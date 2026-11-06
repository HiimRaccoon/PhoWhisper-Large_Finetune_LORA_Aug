"""Explicit optimizer windows and truthful successful-call progress."""
from dataclasses import dataclass
import math
import time
import os
import socket

import numpy as np
import torch

from .contracts import TrainingBudget
from .precision import autocast_context,require_gpu_allocation


@dataclass
class WindowResult:
    loss: float
    scaled_loss: float
    lr: float
    gradient_norm: float
    window_size: int
    optimizer_step_invoked: bool
    elapsed_seconds: float
    gpu_memory: dict


def compute_budget(n: int) -> TrainingBudget:
    if type(n) is not int or n <= 0:
        raise ValueError('utterance count must be positive')
    updates = math.ceil(n/16)
    total = updates*10
    return TrainingBudget(n,updates,total,math.ceil(total*0.05))


def epoch_order(n: int, epoch: int, seed: int = 42) -> list[int]:
    generator = torch.Generator(device='cpu').manual_seed(seed+epoch)
    return torch.randperm(n,generator=generator).tolist()


def create_optimizer(bundle,config) -> torch.optim.AdamW:
    parameters = []
    for name,parameter in bundle.model.named_parameters():
        if parameter.requires_grad:
            if ('lora_A' not in name and 'lora_B' not in name) or parameter.dtype != torch.float32:
                raise ValueError('optimizer may contain only FP32 LoRA A/B tensors')
            parameters.append(parameter)
    if not parameters:
        raise ValueError('no trainable adapter parameters')
    t = config.training
    return torch.optim.AdamW(parameters,lr=t.learning_rate,betas=t.betas,eps=t.epsilon,
                            weight_decay=t.weight_decay)


def run_window(bundle,optimizer,scheduler,scaler,examples,progress) -> WindowResult:
    if not 1 <= len(examples) <= 16 or progress.pending_micro_steps or not 0 <= progress.epoch < 10:
        raise ValueError('invalid or pending accumulation window')
    profile = bundle.runtime
    if profile.device == 'cuda':
        require_gpu_allocation(os.environ,socket.gethostname())
    if profile.scaler_enabled and scaler is None:
        raise ValueError('FP16 requires GradScaler')
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    raw_losses = []
    k = len(examples)
    lr = optimizer.param_groups[0]['lr']
    invoked = False
    def successful_step(*args,**kwargs):
        nonlocal invoked
        invoked = True
    hook = optimizer.register_step_post_hook(successful_step)
    try:
        bundle.model.train()
        for example in examples:
            inputs = {key:value.to(profile.device) for key,value in example.model_inputs().items()}
            with autocast_context(profile):
                raw_loss = bundle.model(**inputs).loss
            if not torch.isfinite(raw_loss).all():
                raise FloatingPointError('non-finite training loss')
            raw_losses.append(float(raw_loss.detach()))
            loss = raw_loss/k
            (scaler.scale(loss) if scaler is not None else loss).backward()
            progress.pending_micro_steps += 1
        if scaler is not None:
            scaler.unscale_(optimizer)
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('non-finite or disconnected adapter gradients')
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if not invoked:
            raise RuntimeError('GradScaler skipped optimizer update; training stopped')
        scheduler.step()
        progress.global_update += 1
        progress.global_micro_step += k
        progress.next_position += k
        progress.successful_updates_by_epoch[progress.epoch] += 1
        progress.utterances_completed_by_epoch[progress.epoch] += k
        memory = ({'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                   'peak':torch.cuda.max_memory_allocated()} if profile.device == 'cuda'
                  else {'allocated':0,'reserved':0,'peak':0})
        return WindowResult(float(np.mean(raw_losses)),float(np.mean(raw_losses))/k,lr,float(norm),k,
                            invoked,time.perf_counter()-start,memory)
    finally:
        hook.remove()
        optimizer.zero_grad(set_to_none=True)
        progress.pending_micro_steps = 0
