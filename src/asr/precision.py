"""Allocation guards and measured autocast selection, without OOM-driven method changes."""
from contextlib import nullcontext
import gc
import os
import socket

import torch

from .contracts import RuntimeProfile


def require_gpu_allocation(env, hostname: str) -> None:
    host = (hostname or socket.gethostname()).lower().split('.')[0]
    if host.startswith('login'):
        raise RuntimeError('GPU execution is forbidden on the login node')
    if not str(env.get('SLURM_JOB_ID','')).strip():
        raise RuntimeError('GPU execution requires a Slurm compute allocation')


def autocast_context(profile: RuntimeProfile):
    if profile.compute_dtype == 'float32':
        return nullcontext()
    return torch.autocast(device_type=profile.device,dtype=getattr(torch,profile.compute_dtype))


def select_precision(bf16_supported: bool, probe) -> RuntimeProfile:
    failures = []
    for dtype in (['bfloat16','float16'] if bf16_supported else ['float16']):
        try:
            metrics = probe(dtype)
            return RuntimeProfile('cuda',dtype,dtype=='float16',{'failures':failures,'metrics':metrics})
        except torch.OutOfMemoryError:
            raise
        except (RuntimeError,FloatingPointError) as exc:
            if 'out of memory' in str(exc).lower():
                raise
            failures.append({'dtype':dtype,'error':str(exc)})
    raise RuntimeError(f'GPU precision/kernel probes failed: {failures}')


def probe_runtime(config,snapshot) -> RuntimeProfile:
    require_gpu_allocation(os.environ,socket.gethostname())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one allocated CUDA/MIG device is required')
    from .model import initialize_adapter,load_base
    def probe(dtype):
        model = optimizer = scaler = output = None
        try:
            profile = RuntimeProfile('cuda',dtype,dtype=='float16')
            bundle = initialize_adapter(load_base(snapshot),config,profile)
            model = bundle.model.to('cuda').train()
            torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,
                                          betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
            scaler = torch.amp.GradScaler('cuda') if profile.scaler_enabled else None
            with autocast_context(profile):
                output = model(input_features=torch.zeros(1,80,3000,device='cuda'),
                               attention_mask=torch.ones(1,3000,dtype=torch.long,device='cuda'),
                               labels=torch.ones(1,447,dtype=torch.long,device='cuda'))
            if not torch.isfinite(output.loss):
                raise FloatingPointError('non-finite probe loss')
            (scaler.scale(output.loss) if scaler else output.loss).backward()
            if scaler:
                scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('non-finite or disconnected probe gradients')
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            torch.cuda.synchronize()
            return {'gpu':torch.cuda.get_device_name(0),'total_bytes':torch.cuda.get_device_properties(0).total_memory,
                    'peak_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved(),
                    'torch':torch.__version__,'cuda':torch.version.cuda,'targets':len(bundle.target_names),
                    'trainable_parameters':sum(p.numel() for p in parameters)}
        finally:
            del output,scaler,optimizer,model
            if 'bundle' in locals(): del bundle
            gc.collect()
            torch.cuda.empty_cache()
    return select_precision(torch.cuda.is_bf16_supported(),probe)
