import pytest
import torch


def test_smoke_harness_preserves_three_windows_and_resumes_real_optimizer_state(local_config):
    from training_helpers import ToyLoss,examples,bundle_for
    from test_checkpoint import identity_fixture
    from asr.checkpoint import TrainingObjects
    from asr.model import seed_all
    from asr.training import create_optimizer,compute_budget
    from asr.schedule import create_scheduler
    from asr.smoke import compare_resume
    def factory():
        seed_all(42)
        bundle=bundle_for(ToyLoss()); opt=create_optimizer(bundle,local_config)
        return TrainingObjects(bundle,opt,create_scheduler(opt,compute_budget(5036)),None)
    result=compare_resume(factory,examples(44),identity_fixture(),local_config,
                          local_config.paths.project_root/'runs/smoke-test',list(range(5036)))
    assert result['matched'] is True
    assert result['window_sizes']==[16,16,12]
    assert result['global_update']==3 and result['global_micro_step']==44
    assert result['initial_adapter_hash']==result['resumed_initial_adapter_hash']
    assert result['continuous_repeat_matched']
    assert result['immediate_restore_exact']
    assert result['pre_step_gradients_matched']


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
def test_real_whisper_dropout_checkpointing_resume_and_gradients_on_cpu(local_config,tiny_whisper,dtype):
    """Small architecture; real PEFT dropout, non-reentrant recompute and AdamW.

    CPU BF16 exercises autocast semantics, not H200 kernels or GPU acceptance.
    """
    from transformers import WhisperConfig,WhisperForConditionalGeneration
    from asr.checkpoint import TrainingObjects
    from asr.contracts import RuntimeProfile,PreparedExample
    from asr.model import initialize_adapter,seed_all
    from asr.training import create_optimizer,compute_budget
    from asr.schedule import create_scheduler
    from asr.smoke import compare_resume
    from asr.paths import read_json
    from test_checkpoint import identity_fixture
    model_config = tiny_whisper.config.to_dict()
    model_config['max_source_positions'] = 8
    def factory():
        seed_all(123)
        base = WhisperForConditionalGeneration(WhisperConfig(**model_config))
        bundle = initialize_adapter(base,local_config,RuntimeProfile('cpu',dtype,False))
        assert bundle.model.is_gradient_checkpointing
        assert bundle.model.base_model.model.model.encoder._gradient_checkpointing_func.keywords == {
            'use_reentrant':False,'preserve_rng_state':True}
        dropout = [m for n,m in bundle.model.named_modules() if 'lora_dropout' in n and isinstance(m,torch.nn.Dropout)]
        assert len(dropout) == 6 and all(m.p == 0.05 for m in dropout)
        opt = create_optimizer(bundle,local_config)
        return TrainingObjects(bundle,opt,create_scheduler(opt,compute_budget(5036)),None)
    generator = torch.Generator().manual_seed(7)
    prepared = [PreparedExample(torch.randn(1,80,16,generator=generator),torch.ones(1,16,dtype=torch.long),
                                torch.tensor([[3,4,2]]),f'fixture/{i}') for i in range(44)]
    # The laptop's oneDNN cannot execute BF16 convolution backward. Exercise
    # native CPU kernels for BOTH branches; GPU execution keeps its own backend.
    with torch.backends.mkldnn.flags(enabled=False):
        result = compare_resume(factory,prepared,identity_fixture(),local_config,
                                local_config.paths.project_root/'runs'/f'smoke-{dtype}',list(range(5036)))
    assert result['matched'] and result['pre_step_gradients_matched'] and result['immediate_restore_exact']
    assert result['window_sizes'] == [16,16,12] and result['global_update'] == 3
    assert result['rtol'] == 1e-4 and result['atol'] == 1e-5
    diagnostic = read_json(result['diagnostic'])
    assert [len(diagnostic['branches'][name]) for name in ['continuous','continuous_repeat','resumed']] == [3,3,2]
    assert len(diagnostic['optimizer_parameter_names'][0]) == 12
    first = diagnostic['branches']['resumed'][0]
    assert first['window'] == 2 and len(first['losses']) == 16
    assert first['micro_steps'][0]['utterance_id'] == 'fixture/16'
    assert first['metrics']['window_size'] == 16 and first['metrics']['gradient_norm'] > 0


def test_comparison_reports_tensor_name_without_relaxing_tolerance():
    from asr.smoke import _compare
    with pytest.raises(AssertionError,match='optimizer/encoder.lora_B/exp_avg'):
        _compare({'encoder.lora_B':{'exp_avg':torch.zeros(2)}},
                 {'encoder.lora_B':{'exp_avg':torch.ones(2)*0.001}},'optimizer')


def test_continuous_repeat_divergence_is_caught_before_resume(local_config):
    from training_helpers import ToyLoss,examples,bundle_for
    from test_checkpoint import identity_fixture
    from asr.checkpoint import TrainingObjects
    from asr.model import seed_all
    from asr.training import create_optimizer,compute_budget
    from asr.schedule import create_scheduler
    from asr.smoke import compare_resume
    from asr.paths import read_json
    calls = []
    def factory():
        seed_all(42)
        bundle = bundle_for(ToyLoss())
        calls.append(bundle)
        if len(calls) == 2:
            bundle.model.lora_A.register_hook(lambda gradient:gradient+0.1)
        opt = create_optimizer(bundle,local_config)
        return TrainingObjects(bundle,opt,create_scheduler(opt,compute_budget(5036)),None)
    run = local_config.paths.project_root/'runs/smoke-divergence'
    with pytest.raises(AssertionError,match='continuous_repeat/window-1/before_optimizer/gradients/lora_A'):
        compare_resume(factory,examples(44),identity_fixture(),local_config,run,list(range(5036)))
    diagnostic = read_json(run/'resume-diagnostic.json')
    assert len(calls) == 2 and 'resumed' not in diagnostic['branches']
    assert len(diagnostic['branches']['continuous_repeat']) == 1


def test_smoke_entry_refuses_cpu_fixture_as_actual_gpu_evidence(local_config,monkeypatch):
    from asr.smoke import run_smoke
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='allocation'):
        run_smoke(local_config,None,[],'smoke-refused')
    assert not (local_config.paths.project_root/'runs/smoke-refused').exists()


def test_smoke_cli_reports_missing_evidence_with_nonzero_exit(tmp_path):
    from asr.cli import main
    assert main(['smoke','--run-id','smoke-cli','--cpu-report',str(tmp_path/'cpu.json'),
                 '--gpu-report',str(tmp_path/'gpu.json')])==1
