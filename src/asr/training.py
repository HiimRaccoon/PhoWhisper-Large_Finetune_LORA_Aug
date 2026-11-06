"""Explicit optimizer windows and truthful successful-call progress."""
from dataclasses import dataclass, field
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
    gradient_evidence: dict = field(default_factory=dict)


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


def run_window(bundle,optimizer,scheduler,scaler,examples,progress,collect_gradient_evidence=False) -> WindowResult:
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
        evidence = {}
        if collect_gradient_evidence:
            groups = set()
            for name,parameter in bundle.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None and parameter.grad.abs().sum().item() > 0:
                    side = 'encoder' if 'encoder.layers.' in name else 'decoder' if 'decoder.layers.' in name else 'other'
                    groups.add(side+('_lora_A' if 'lora_A' in name else '_lora_B'))
            evidence = {'nonzero_groups':sorted(groups),'all_connected':True,'finite':True,
                        'frozen_gradients':sum(p.grad is not None for p in bundle.model.parameters() if not p.requires_grad)}
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
                            invoked,time.perf_counter()-start,memory,evidence)
    finally:
        hook.remove()
        optimizer.zero_grad(set_to_none=True)
        progress.pending_micro_steps = 0


def train_loop(objects,rows,prepare,config,run_dir,attempt_id,identity,progress=None,permutation=None,
               event_callback=None,stop_requested=None):
    """The same epoch/window driver serves official execution and tiny CPU integration tests."""
    from dataclasses import asdict
    from .checkpoint import capture_training,save_checkpoint
    from .config import canonical_hash
    from .contracts import TrainingProgress
    from .paths import project_path
    budget = compute_budget(len(rows))
    run_dir = project_path(config,run_dir)
    progress = progress or TrainingProgress()
    emit = event_callback or (lambda record:None)
    stop = stop_requested or (lambda:False)
    if progress.completed_epochs == 10:
        return progress
    if progress.epoch != progress.completed_epochs or progress.pending_micro_steps or progress.next_position % 16:
        raise ValueError('invalid resume epoch/window boundary')
    order = list(permutation) if permutation is not None else epoch_order(len(rows),progress.epoch)
    if order != epoch_order(len(rows),progress.epoch):
        raise ValueError('resume permutation does not match epoch seed')
    digest = canonical_hash(order)
    if progress.permutation_hash and progress.permutation_hash != digest:
        raise ValueError('resume permutation hash mismatch')
    progress.permutation_hash = digest

    def checkpoint(kind,reasons):
        state = capture_training(objects,progress,identity,order,kind,config)
        state.metadata.update(reasons=reasons,budget=asdict(budget))
        path = save_checkpoint(run_dir,attempt_id,state)
        emit({'event':'checkpoint','path':str(path),'kind':kind,'reasons':reasons,
              'global_update':progress.global_update,'completed_epochs':progress.completed_epochs})

    while progress.epoch < 10:
        if stop():
            checkpoint('interrupted',['interrupted'])
            return progress
        epoch = progress.epoch
        for start in range(progress.next_position,len(rows),16):
            batch = [prepare(rows[index],epoch) for index in order[start:start+16]]
            result = run_window(objects.bundle,objects.optimizer,objects.scheduler,objects.scaler,batch,progress)
            emit({'event':'optimizer_update','epoch':epoch,'position':progress.next_position,
                  'micro_step':progress.global_micro_step,'global_update':progress.global_update,
                  'loss':result.loss,'scaled_loss':result.scaled_loss,'window_size':result.window_size,'lr':result.lr,
                  'gradient_norm':result.gradient_norm,'dtype':objects.bundle.runtime.compute_dtype,
                  'scaler':{'enabled':objects.scaler is not None,
                            'scale':objects.scaler.get_scale() if objects.scaler is not None else None},
                  'gpu_memory':result.gpu_memory,'elapsed_seconds':result.elapsed_seconds,
                  'throughput':result.window_size/max(result.elapsed_seconds,1e-12)})
            reasons = []
            if progress.global_update % 50 == 0:
                reasons.append('interval')
            at_end = progress.next_position == len(rows)
            if at_end:
                if progress.successful_updates_by_epoch[epoch] != budget.updates_per_epoch or progress.utterances_completed_by_epoch[epoch] != len(rows):
                    raise ValueError('completed epoch has incorrect row/update coverage')
                progress.completed_epochs += 1
                progress.epoch += 1
                progress.next_position = 0
                reasons.append('epoch')
                if progress.completed_epochs == 10:
                    reasons.append('terminal')
                else:
                    order = epoch_order(len(rows),progress.epoch)
                    progress.permutation_hash = canonical_hash(order)
            stopping = stop() and progress.completed_epochs < 10
            if stopping:
                reasons.append('interrupted')
            if reasons:
                kind = ('terminal' if 'terminal' in reasons else 'interrupted' if stopping else
                        'epoch' if at_end else 'regular')
                checkpoint(kind,reasons)
            if stopping or progress.completed_epochs == 10:
                return progress
            if at_end:
                break
    return progress


def execute_run(config,run_dir,resume_from=None):
    """Official CUDA entry: admission and input identity are revalidated before model creation."""
    from pathlib import Path
    import signal
    import threading
    import uuid
    from filelock import FileLock
    from .artifacts import start_attempt,set_status
    from .assets import verify_snapshot,verify_noise
    from .augmentation import augment
    from .checkpoint import TrainingObjects,load_checkpoint,restore_training,discover_checkpoints
    from .contracts import ArtifactIdentity,PreflightReport,SmokeReport
    from .logging import RunLogger
    from .model import build_training_model
    from .paths import project_path,read_json,read_jsonl,sha256_file,write_json_atomic
    from .preflight import check_admission,load_audited_rows
    from .processor import load_processor,prepare_example
    from .data import read_pcm16
    from .paths import audio_path
    from .schedule import create_scheduler

    require_gpu_allocation(os.environ,socket.gethostname())
    run_dir = project_path(config,Path(run_dir))
    reports = [PreflightReport.from_dict(read_json(run_dir/'preflight'/f'{stage}.json')) for stage in ['cpu','gpu']]
    smoke = SmokeReport.from_dict(read_json(run_dir/'preflight/smoke.json'))
    identity = check_admission(config,reports,smoke)
    manifest = read_json(run_dir/'run-manifest.json')
    if manifest.get('mode') == 'smoke':
        raise ValueError('smoke runs cannot be resumed as official training')
    if manifest['identity'] != identity.as_dict():
        raise ValueError('run identity changed')
    frozen_manifest = run_dir/'manifests/utterances.jsonl'
    if sha256_file(frozen_manifest) != identity.manifest_hash:
        raise ValueError('run manifest copy hash mismatch')
    if resume_from is not None:
        resume_from = project_path(config,Path(resume_from))
        if not resume_from.is_relative_to((run_dir/'checkpoints').resolve()):
            raise ValueError('resume checkpoint belongs to another run')
    runtime = next(r.runtime for r in reports if r.stage=='gpu')
    with FileLock(str(run_dir/'.session.lock'),timeout=0):
        attempt = start_attempt(run_dir)
        set_status(run_dir,'running',latest_attempt=attempt)
        stop = threading.Event()
        old_handlers = {}
        for name in ['SIGTERM','SIGINT','SIGUSR1']:
            signum = getattr(signal,name,None)
            if signum is not None:
                old_handlers[signum] = signal.signal(signum,lambda *_:stop.set())
        with RunLogger(run_dir/'attempts'/attempt/'logs/events.jsonl',run_dir.name,attempt) as logger:
            try:
                rows,_ = load_audited_rows(config,verify_audio=False)
                snapshot,noise = verify_snapshot(config),verify_noise(config)
                processor = load_processor(snapshot)
                bundle = build_training_model(config,snapshot,runtime)
                record = read_json(run_dir/'run-manifest.json')
                record.update(lora_targets=bundle.target_names,
                              trainable_parameters=sum(p.numel() for p in bundle.model.parameters() if p.requires_grad),
                              base_parameters=sum(p.numel() for p in bundle.model.parameters() if not p.requires_grad))
                write_json_atomic(run_dir/'run-manifest.json',record)
                optimizer = create_optimizer(bundle,config)
                scheduler = create_scheduler(optimizer,compute_budget(len(rows)))
                scaler = torch.amp.GradScaler('cuda') if runtime.scaler_enabled else None
                objects = TrainingObjects(bundle,optimizer,scheduler,scaler)
                progress,permutation = None,None
                if resume_from is not None:
                    state = load_checkpoint(resume_from,identity)
                    if state.metadata.get('run_id') != run_dir.name or state.metadata.get('kind')=='smoke':
                        raise ValueError('checkpoint cannot initialize this official run')
                    progress = restore_training(state,objects)
                    permutation = state.permutation
                    logger.log('resume',checkpoint=str(resume_from),restored_update=progress.global_update)
                def prepare(row,epoch):
                    audio,metadata = read_pcm16(audio_path(config.paths.train_root,row.audio))
                    if metadata['wav_sha256'] != row.wav_sha256:
                        raise ValueError(f'audio changed during training: {row.utterance_id}')
                    view = augment(audio,row,epoch,noise,config.training.seed)
                    logger.log('augmentation',**view.as_log())
                    return prepare_example(row,view.audio,processor)
                def event(record):
                    payload = dict(record); name = payload.pop('event')
                    logger.log(name,**payload)
                progress = train_loop(objects,rows,prepare,config,run_dir,attempt,identity,
                                      progress,permutation,event,stop.is_set)
                if progress.completed_epochs < 10:
                    set_status(run_dir,'interrupted',progress=progress.as_dict())
                else:
                    final_checkpoint = discover_checkpoints(run_dir)[-1]
                    final = run_dir/'final-adapter'
                    origin = {'schema_version':1,'identity':identity.as_dict(),'checkpoint':str(final_checkpoint),
                              'checkpoint_adapter_sha256':sha256_file(final_checkpoint/'adapter.safetensors')}
                    if final.exists():
                        if read_json(final/'adapter-origin.json') != origin:
                            raise FileExistsError('existing final adapter identity differs')
                    else:
                        temporary = run_dir/f'.tmp-final-adapter-{uuid.uuid4().hex}'
                        bundle.model.save_pretrained(str(temporary),safe_serialization=True)
                        write_json_atomic(temporary/'adapter-origin.json',origin)
                        os.rename(temporary,final)
                    set_status(run_dir,'training_complete',progress=progress.as_dict(),final_checkpoint=str(final_checkpoint))
                write_json_atomic(run_dir/'run-summary.json',{'schema_version':1,'identity':identity.as_dict(),
                                  'status':read_json(run_dir/'run-manifest.json')['status'],'progress':progress.as_dict(),
                                  'latest_attempt':attempt,'latest_attempt_augmentation':logger.summary})
                return progress
            except Exception as exc:
                logger.log('failure',error=f'{type(exc).__name__}: {exc}')
                set_status(run_dir,'failed',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                for signum,handler in old_handlers.items():
                    signal.signal(signum,handler)
