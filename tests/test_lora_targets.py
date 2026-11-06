import pytest


def test_exact_target_topology_has_192_names_without_allocating_large_weights():
    import torch
    from transformers import WhisperConfig,WhisperForConditionalGeneration
    from asr.model import resolve_lora_targets
    config=WhisperConfig(encoder_layers=32,decoder_layers=32,d_model=1280,
                         encoder_attention_heads=20,decoder_attention_heads=20,
                         encoder_ffn_dim=5120,decoder_ffn_dim=5120)
    with torch.device('meta'):
        model=WhisperForConditionalGeneration(config)
        model.unrelated=torch.nn.Linear(1280,1280)
    names=resolve_lora_targets(model)
    assert len(names)==192
    assert 'model.encoder.layers.31.self_attn.v_proj' in names
    assert 'model.decoder.layers.31.encoder_attn.q_proj' in names
    assert all('unrelated' not in name for name in names)
    del model.model.encoder.layers[0].self_attn.q_proj
    with pytest.raises(ValueError,match='q_proj'):
        resolve_lora_targets(model)


def test_adapter_wrapper_receives_input_features_and_only_ab_train(local_config,tiny_whisper):
    from asr.model import initialize_adapter
    bundle=initialize_adapter(tiny_whisper,local_config)
    trained=[(name,p) for name,p in bundle.model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _,p in trained)==1536
    assert all('lora_A' in n or 'lora_B' in n for n,_ in trained)
    assert len(bundle.target_names)==6


def test_checkpointed_gradient_reaches_encoder_decoder_and_base_is_frozen(local_config,tiny_whisper):
    import torch
    from asr.model import initialize_adapter
    bundle=initialize_adapter(tiny_whisper,local_config)
    model=bundle.model.train()
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    features=torch.randn(1,80,3000)
    labels=torch.tensor([[3,4,2]])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=model(input_features=features,attention_mask=torch.ones(1,3000,dtype=torch.long),labels=labels).loss
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
    assert any('encoder.layers' in n and 'lora_A' in n and g.abs().sum()>0 for n,g in grads.items())
    assert any('decoder.layers' in n and 'lora_A' in n and g.abs().sum()>0 for n,g in grads.items())
    assert all(torch.equal(p,frozen[n]) and p.grad is None for n,p in model.named_parameters() if n in frozen)


def test_window_can_report_gradient_connectivity_after_nonzero_adapter_update(local_config,tiny_whisper):
    import torch
    from asr.model import initialize_adapter
    from asr.training import create_optimizer,compute_budget,run_window
    from asr.schedule import create_scheduler
    from asr.contracts import PreparedExample,TrainingProgress
    bundle=initialize_adapter(tiny_whisper,local_config)
    opt=create_optimizer(bundle,local_config); sched=create_scheduler(opt,compute_budget(5036))
    example=PreparedExample(torch.randn(1,80,3000),torch.ones(1,3000,dtype=torch.long),torch.tensor([[3,4,2]]),'s/current')
    progress=TrainingProgress()
    for _ in range(3): result=run_window(bundle,opt,sched,None,[example],progress,collect_gradient_evidence=True)
    assert 'encoder_lora_A' in result.gradient_evidence['nonzero_groups']
    assert 'decoder_lora_A' in result.gradient_evidence['nonzero_groups']
    assert result.gradient_evidence['frozen_gradients']==0
