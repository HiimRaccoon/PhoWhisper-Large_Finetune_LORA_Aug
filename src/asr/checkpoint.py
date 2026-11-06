"""Immutable complete checkpoints; trusted tensor-only deserialization and RNG-last restore."""
from __future__ import annotations

import copy
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import re
import uuid

import numpy as np
import torch
from filelock import FileLock,Timeout
from safetensors.torch import load_file,save as serialize_tensors

from .config import canonical_hash
from .contracts import ArtifactIdentity,TrainingProgress
from .paths import atomic_bytes,read_json,resolve_beneath,sha256_file,write_json_atomic


@dataclass
class TrainingObjects:
    bundle: object
    optimizer: object
    scheduler: object
    scaler: object | None


@dataclass
class CheckpointState:
    identity: ArtifactIdentity
    progress: TrainingProgress
    permutation: list[int]
    adapter: dict
    optimizer: dict
    scheduler: dict
    scaler: dict | None
    rng: dict
    metadata: dict


def capture_rng(cuda: bool = False) -> dict:
    state = np.random.get_state()
    return {'python':random.getstate(),'numpy':{'algorithm':state[0],'keys':state[1].tolist(),
             'position':state[2],'has_gauss':state[3],'cached_gaussian':state[4]},
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if cuda else []}


def restore_rng(rng: dict) -> None:
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])


def adapter_state(model) -> dict:
    return {name:p.detach().cpu().clone().contiguous() for name,p in model.named_parameters() if p.requires_grad}


def restore_adapter(model,weights: dict) -> None:
    parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != weights.keys():
        raise ValueError('checkpoint adapter target names mismatch')
    for name,parameter in parameters.items():
        value = weights[name]
        if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'invalid checkpoint adapter tensor: {name}')
    with torch.no_grad():
        for name,parameter in parameters.items():
            parameter.copy_(weights[name].to(parameter.device))


def capture_training(objects,progress,identity,permutation,kind='regular',config=None) -> CheckpointState:
    if progress.pending_micro_steps or any(p.grad is not None for p in objects.bundle.model.parameters()):
        raise ValueError('cannot checkpoint pending accumulation or gradients')
    if progress.permutation_hash != canonical_hash(permutation):
        raise ValueError('permutation hash mismatch')
    peft = getattr(objects.bundle.model,'peft_config',{})
    adapter_config = peft['default'].to_dict() if 'default' in peft else {}
    adapter_config = json.loads(json.dumps(adapter_config,default=lambda v:sorted(v) if isinstance(v,set) else str(v)))
    metadata = {'kind':kind,'runtime':objects.bundle.runtime.as_dict(),'target_names':objects.bundle.target_names,
                'trainable_count':sum(p.numel() for p in objects.bundle.model.parameters() if p.requires_grad),
                'lora_config':adapter_config,'resolved_config':config.resolved_dict() if config else {},
                'pending_gradient_count':0,'created_at':datetime.now(timezone.utc).isoformat()}
    return CheckpointState(identity,copy.deepcopy(progress),list(permutation),adapter_state(objects.bundle.model),
            copy.deepcopy(objects.optimizer.state_dict()),copy.deepcopy(objects.scheduler.state_dict()),
            copy.deepcopy(objects.scaler.state_dict()) if objects.scaler is not None else None,
            capture_rng(objects.bundle.runtime.device=='cuda'),metadata)


def _boundary(state):
    p = state.progress
    if p.pending_micro_steps or state.metadata.get('pending_gradient_count') != 0:
        raise ValueError('checkpoint has pending accumulation/gradients')
    if len(p.successful_updates_by_epoch) != 10 or len(p.utterances_completed_by_epoch) != 10:
        raise ValueError('invalid per-epoch progress')
    if sum(p.successful_updates_by_epoch) != p.global_update or sum(p.utterances_completed_by_epoch) != p.global_micro_step:
        raise ValueError('checkpoint progress counters disagree')
    if p.permutation_hash != canonical_hash(state.permutation):
        raise ValueError('checkpoint permutation hash mismatch')


def save_checkpoint(run_dir: Path,attempt_id: str,state: CheckpointState) -> Path:
    _boundary(state)
    kind = state.metadata.get('kind','regular')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',attempt_id) or not re.fullmatch(r'[a-z_]+',kind):
        raise ValueError('unsafe checkpoint attempt/kind')
    attempt = resolve_beneath(Path(run_dir),'checkpoints/'+attempt_id)
    attempt.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(attempt/'.writer.lock'),timeout=0):
            destination = attempt/f'update-{state.progress.global_update:06d}-{kind}'
            if destination.exists():
                raise FileExistsError(destination)
            temporary = attempt/f'.tmp-checkpoint-{uuid.uuid4().hex}'
            temporary.mkdir()
            atomic_bytes(temporary/'adapter.safetensors',serialize_tensors(state.adapter))
            with (temporary/'state.pt').open('xb') as stream:
                torch.save({'optimizer':state.optimizer,'scheduler':state.scheduler,'scaler':state.scaler,'rng':state.rng},stream)
                stream.flush(); os.fsync(stream.fileno())
            metadata = {**state.metadata,'schema_version':1,'run_id':Path(run_dir).name,'identity':state.identity.as_dict(),
                        'progress':state.progress.as_dict(),'permutation':state.permutation,
                        'adapter_sha256':sha256_file(temporary/'adapter.safetensors')}
            write_json_atomic(temporary/'metadata.json',metadata)
            files = {name:sha256_file(temporary/name) for name in ['adapter.safetensors','state.pt','metadata.json']}
            write_json_atomic(temporary/'COMPLETE.json',{'schema_version':1,'files':files})
            os.rename(temporary,destination)
            return destination
    except Timeout as exc:
        raise RuntimeError('another checkpoint writer owns this attempt') from exc


def load_checkpoint(path: Path,expected: ArtifactIdentity) -> CheckpointState:
    path = Path(path)
    if path.name.startswith('.tmp') or not (path/'COMPLETE.json').is_file():
        raise ValueError('incomplete checkpoint')
    manifest = read_json(path/'COMPLETE.json')
    if manifest.get('schema_version') != 1 or set(manifest.get('files',{})) != {'adapter.safetensors','state.pt','metadata.json'}:
        raise ValueError('invalid checkpoint completion manifest')
    for name,digest in manifest['files'].items():
        if sha256_file(resolve_beneath(path,name)) != digest:
            raise ValueError(f'checkpoint hash mismatch: {name}')
    metadata = read_json(path/'metadata.json')
    if metadata.get('identity') != expected.as_dict():
        raise ValueError('checkpoint identity mismatch')
    if metadata.get('adapter_sha256') != manifest['files']['adapter.safetensors']:
        raise ValueError('adapter identity/hash mismatch')
    payload = torch.load(path/'state.pt',map_location='cpu',weights_only=True)
    state = CheckpointState(expected,TrainingProgress(**metadata['progress']),metadata['permutation'],
                            load_file(str(path/'adapter.safetensors')),payload['optimizer'],payload['scheduler'],
                            payload['scaler'],payload['rng'],metadata)
    _boundary(state)
    return state


def restore_training(state: CheckpointState,objects: TrainingObjects) -> TrainingProgress:
    _boundary(state)
    if state.metadata['runtime']['compute_dtype'] != objects.bundle.runtime.compute_dtype:
        raise ValueError('checkpoint runtime precision mismatch')
    if (state.scaler is None) != (objects.scaler is None):
        raise ValueError('checkpoint scaler profile mismatch')
    restore_adapter(objects.bundle.model,state.adapter)
    objects.optimizer.load_state_dict(state.optimizer)
    objects.scheduler.load_state_dict(state.scheduler)
    if objects.scaler is not None:
        objects.scaler.load_state_dict(state.scaler)
    restore_rng(state.rng)
    return copy.deepcopy(state.progress)


def discover_checkpoints(run_dir: Path) -> list[Path]:
    root = Path(run_dir)/'checkpoints'
    found = []
    if not root.exists():
        return found
    for marker in root.rglob('COMPLETE.json'):
        if any(part.startswith('.tmp') for part in marker.parts):
            continue
        directory = resolve_beneath(root,marker.parent.relative_to(root).as_posix())
        try:
            metadata = read_json(directory/'metadata.json')
            progress = metadata['progress']
            found.append(((progress['global_update'],progress['completed_epochs'],metadata['kind']=='terminal',str(directory)),directory))
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return [directory for _,directory in sorted(found)]
