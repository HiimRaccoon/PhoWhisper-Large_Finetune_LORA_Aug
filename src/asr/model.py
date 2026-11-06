"""Exact-name LoRA injection; official model loading requires a verified local snapshot."""
from dataclasses import dataclass
import random
import numpy as np
import torch

from .contracts import RuntimeProfile


@dataclass
class TrainingBundle:
    model: torch.nn.Module
    target_names: list[str]
    runtime: RuntimeProfile


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_lora_targets(model: torch.nn.Module) -> list[str]:
    config = model.config
    names = []
    for side,count in [('encoder',config.encoder_layers),('decoder',config.decoder_layers)]:
        for layer in range(count):
            for attention in (['self_attn'] if side == 'encoder' else ['self_attn','encoder_attn']):
                for projection in ['q_proj','v_proj']:
                    name = f'model.{side}.layers.{layer}.{attention}.{projection}'
                    try:
                        module = model.get_submodule(name)
                    except AttributeError as exc:
                        raise ValueError(f'missing LoRA target {name}') from exc
                    if not isinstance(module,torch.nn.Linear) or (module.in_features,module.out_features) != (config.d_model,config.d_model):
                        raise ValueError(f'incorrect LoRA target shape/type: {name}')
                    names.append(name)
    return names


def initialize_adapter(base: torch.nn.Module, config, runtime=None) -> TrainingBundle:
    """Shared initialization seam; tiny CPU fixtures exercise the same adapter mechanism."""
    from peft import LoraConfig,get_peft_model
    seed_all(config.training.seed)
    base.float().requires_grad_(False)
    names = resolve_lora_targets(base)
    adapter = LoraConfig(r=config.lora.rank,lora_alpha=config.lora.alpha,lora_dropout=config.lora.dropout,
                         bias='none',modules_to_save=None,target_modules=names,task_type=None)
    model = get_peft_model(base,adapter)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_A' not in n and 'lora_B' not in n for n,_ in trainable):
        raise ValueError('only LoRA A/B may be trainable')
    for _,parameter in trainable:
        if parameter.dtype != torch.float32:
            raise ValueError('LoRA storage must remain FP32')
    attached = {n.removeprefix('base_model.model.') for n,m in model.named_modules() if hasattr(m,'lora_A')}
    if attached != set(names):
        raise ValueError('actual adapter target set differs from resolved targets')
    return TrainingBundle(model,names,runtime or RuntimeProfile())


def load_base(snapshot):
    from transformers import WhisperForConditionalGeneration
    from .assets import inventory_files
    if inventory_files(snapshot.path) != snapshot.files:
        raise ValueError('snapshot changed since verification')
    model,info = WhisperForConditionalGeneration.from_pretrained(
        str(snapshot.path),local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'pretrained weights did not load exactly: {info}')
    return model


def build_training_model(config, snapshot, runtime: RuntimeProfile) -> TrainingBundle:
    from .precision import require_gpu_allocation
    import os,socket
    if runtime.device != 'cuda':
        raise ValueError('official training requires allocated CUDA runtime')
    require_gpu_allocation(os.environ,socket.gethostname())
    base = load_base(snapshot)
    if (base.config.encoder_layers,base.config.decoder_layers,base.config.d_model,base.config.num_mel_bins) != (32,32,1280,80):
        raise ValueError('official snapshot architecture mismatch')
    bundle = initialize_adapter(base,config,runtime)
    if len(bundle.target_names) != 192 or sum(p.numel() for p in bundle.model.parameters() if p.requires_grad) != 3932160:
        raise ValueError('official LoRA target/parameter count mismatch')
    bundle.model.to(runtime.device)
    return bundle
