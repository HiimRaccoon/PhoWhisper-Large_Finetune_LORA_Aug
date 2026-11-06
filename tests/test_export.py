import copy
from dataclasses import replace
import pytest
import torch


def terminal_fixture(config):
    from test_checkpoint import training_fixture
    from asr.training import compute_budget,epoch_order
    from asr.config import canonical_hash
    objects,state=training_fixture(config)
    budget=compute_budget(5036)
    p=state.progress
    p.epoch=10; p.completed_epochs=10; p.next_position=0; p.global_update=3150; p.global_micro_step=50360
    p.successful_updates_by_epoch=[315]*10; p.utterances_completed_by_epoch=[5036]*10
    state.permutation=epoch_order(5036,9); p.permutation_hash=canonical_hash(state.permutation)
    state.metadata.update(kind='terminal',adapter_sha256='a'*64,run_id='fixture')
    for item in state.optimizer['state'].values(): item['step']=torch.tensor(3150.0)
    state.scheduler['last_epoch']=3150; state.scheduler['_last_lr']=[0.0]
    for group in state.optimizer['param_groups']: group['lr']=0.0
    return state,budget


def test_terminal_guard_accepts_both_finished_epoch_representations(local_config):
    from asr.export import verify_terminal_checkpoint
    state,budget=terminal_fixture(local_config)
    verify_terminal_checkpoint(state,budget,state.identity)
    state.progress.epoch=9; state.progress.next_position=5036
    verify_terminal_checkpoint(state,budget,state.identity)


@pytest.mark.parametrize('field,value',[('completed_epochs',9),('global_update',3149),
                                        ('pending_micro_steps',1),('global_micro_step',50359)])
def test_terminal_guard_rejects_incomplete_training_even_if_named_final(local_config,field,value):
    from asr.export import verify_terminal_checkpoint
    state,budget=terminal_fixture(local_config)
    setattr(state.progress,field,value)
    with pytest.raises(ValueError): verify_terminal_checkpoint(state,budget,state.identity)


def test_terminal_guard_rejects_wrong_adapter_and_optimizer_progress(local_config):
    from asr.export import verify_terminal_checkpoint
    state,budget=terminal_fixture(local_config)
    state.metadata['kind']='epoch'
    with pytest.raises(ValueError): verify_terminal_checkpoint(state,budget,state.identity)
    state.metadata['kind']='terminal'; state.metadata['adapter_sha256']=''
    with pytest.raises(ValueError): verify_terminal_checkpoint(state,budget,state.identity)
    state.metadata['adapter_sha256']='a'*64
    next(iter(state.optimizer['state'].values()))['step']=torch.tensor(1.0)
    with pytest.raises(ValueError): verify_terminal_checkpoint(state,budget,state.identity)


def tiny_export_components(config):
    from transformers import WhisperConfig,WhisperForConditionalGeneration,WhisperFeatureExtractor,WhisperTokenizer,WhisperProcessor
    from transformers.models.whisper.tokenization_whisper import LANGUAGES,bytes_to_unicode
    from asr.model import initialize_adapter
    import json
    root=config.paths.project_root/'tokenizer'; root.mkdir(parents=True,exist_ok=True)
    vocab={f'<|unused_{i}|>':i for i in range(51865)}
    # GPT-2 byte alphabet plus Whisper's true special-token positions.
    for i,char in enumerate(bytes_to_unicode().values()):
        del vocab[f'<|unused_{i}|>']; vocab[char]=i
    specials={'<|endoftext|>':50257,'<|startoftranscript|>':50258}
    specials.update({f'<|{lang}|>':50259+i for i,lang in enumerate(LANGUAGES)})
    specials.update({'<|translate|>':50358,'<|transcribe|>':50359,'<|startoflm|>':50360,
                     '<|startofprev|>':50361,'<|nospeech|>':50362,'<|notimestamps|>':50363})
    for token,i in specials.items():
        vocab.pop(f'<|unused_{i}|>',None); vocab[token]=i
    (root/'vocab.json').write_text(json.dumps(vocab),encoding='utf-8')
    (root/'merges.txt').write_text('#version: 0.2\n',encoding='utf-8')
    tokenizer=WhisperTokenizer(str(root/'vocab.json'),str(root/'merges.txt'),language='vi',task='transcribe',
                               additional_special_tokens=list(specials))
    processor=WhisperProcessor(WhisperFeatureExtractor(feature_size=80),tokenizer)
    model_config=WhisperConfig(vocab_size=51865,d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
        pad_token_id=50257,bos_token_id=50257,eos_token_id=50257,decoder_start_token_id=50258,
        max_target_positions=448,max_source_positions=1500,apply_spec_augment=False,
        suppress_tokens=[],begin_suppress_tokens=[])
    torch.set_num_threads(1)
    bundle=initialize_adapter(WhisperForConditionalGeneration(model_config),config)
    # A nonzero adapter delta makes the numerical merge check meaningful.
    with torch.no_grad():
        for name,p in bundle.model.named_parameters():
            if 'lora_B' in name: p.fill_(0.001)
    return bundle,processor


def test_tiny_model_merge_publishes_reloadable_fp16_artifact(local_config):
    from asr.export import merge_and_export,load_export
    from test_checkpoint import identity_fixture
    bundle,processor=tiny_export_components(local_config)
    output=local_config.paths.project_root/'runs/export/merged-fp16'
    result=merge_and_export(bundle.model,processor,output,identity_fixture(),{'checkpoint':'fixture'},device='cpu')
    assert result.valid and result.metadata['fresh_process_reload']['valid']
    assert result.metadata['comparison']['passed']
    verified=load_export(output)
    assert verified.identity==identity_fixture()
    assert 'model.safetensors' in verified.files
    with pytest.raises(FileExistsError):
        merge_and_export(bundle.model,processor,output,identity_fixture(),{},device='cpu')
    with (output/'config.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError,match='hash'): load_export(output)


def test_cli_export_rejects_missing_checkpoint_before_publication(local_config,capsys):
    from asr.cli import main
    output=local_config.paths.project_root/'runs/final/merged-fp16'
    assert main(['merge-export','--config',str(local_config.paths.project_root/'configs/official.yaml'),
                 '--checkpoint',str(local_config.paths.project_root/'missing'),
                 '--output',str(output)])==1
    assert not output.exists()
    assert 'FileNotFoundError' in capsys.readouterr().err
