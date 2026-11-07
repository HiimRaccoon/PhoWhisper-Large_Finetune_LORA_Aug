"""Isolated three-window proof of forward/backward, optimizer and checkpoint continuation."""
import copy
from dataclasses import asdict
import gc
import hashlib
import os
from pathlib import Path
import socket

import numpy as np
import torch

from .checkpoint import (TrainingObjects,adapter_state,capture_training,load_checkpoint,
                         optimizer_parameter_names,restore_training,save_checkpoint)
from .config import canonical_hash
from .contracts import SmokeReport,TrainingProgress,PreparedExample
from .paths import project_path,read_json,write_json_atomic,atomic_bytes
from .precision import configure_determinism,require_gpu_allocation
from .training import compute_budget,create_optimizer,epoch_order,run_window
from .schedule import create_scheduler


def _cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [_cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _compare(expected,actual,path='state',exact=False):
    if isinstance(expected,torch.Tensor):
        try:
            torch.testing.assert_close(actual.detach().cpu(),expected,
                                       rtol=0 if exact else 1e-4,atol=0 if exact else 1e-5)
        except AssertionError as exc:
            raise AssertionError(f'{path}: {exc}') from exc
    elif isinstance(expected,dict):
        if expected.keys() != actual.keys(): raise AssertionError('state keys differ')
        for key in expected: _compare(expected[key],actual[key],f'{path}/{key}',exact)
    elif isinstance(expected,(list,tuple)):
        if len(expected) != len(actual): raise AssertionError('state lengths differ')
        for index,(first,second) in enumerate(zip(expected,actual)):
            _compare(first,second,f'{path}/{index}',exact)
    elif expected != actual:
        raise AssertionError(f'{path}: state differs: {expected!r} != {actual!r}')


def _weights_hash(weights):
    digest = hashlib.sha256()
    for name,value in sorted(weights.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _frozen_hash(model):
    return _weights_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})


def compare_resume(factory,examples,identity,config,run_dir,permutation,attempt_id='smoke'):
    """Portable harness; its output is not an allocated-GPU SmokeReport."""
    if len(examples) != 44:
        raise ValueError('smoke requires exactly 44 fixture occurrences')
    configure_determinism()
    run_dir = project_path(config,run_dir)
    windows = [examples[:16],examples[16:32],examples[32:]]
    traces = []
    diagnostic = {'branches':{},'rtol':1e-4,'atol':1e-5}
    diagnostic_path = run_dir/'resume-diagnostic.json'

    def observer(branch,index):
        def observe(trace):
            record = {'window':index+1,'losses':trace['losses'].tolist(),'lr':trace['lr'],
                      'micro_steps':trace['micro_steps'],
                      'gradient_hashes':{name:_weights_hash({name:value}) for name,value in trace['gradients'].items()}}
            diagnostic['branches'].setdefault(branch,[]).append(record)
            write_json_atomic(diagnostic_path,diagnostic)
            if branch == 'continuous':
                traces.append(trace)
            else:
                _compare(traces[index],trace,f'{branch}/window-{index+1}/before_optimizer')
        return observe

    def record_result(branch,result):
        diagnostic['branches'][branch][-1]['metrics'] = asdict(result)
        write_json_atomic(diagnostic_path,diagnostic)

    original = factory()
    diagnostic['runtime'] = original.bundle.runtime.as_dict()
    diagnostic['determinism'] = {
        'algorithms':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    initial_hash = _weights_hash(adapter_state(original.bundle.model))
    frozen_before = _frozen_hash(original.bundle.model)
    progress = TrainingProgress(permutation_hash=canonical_hash(permutation))
    results = []
    for index,window in enumerate(windows):
        results.append(run_window(original.bundle,original.optimizer,original.scheduler,original.scaler,
                                   window,progress,collect_gradient_evidence=index==2,
                                   before_step=observer('continuous',index)))
        record_result('continuous',results[-1])
        if index == 0:
            first = capture_training(original,progress,identity,permutation,'smoke',config)
            checkpoint = save_checkpoint(run_dir,attempt_id+'-baseline',first)
            del first
    if _frozen_hash(original.bundle.model) != frozen_before:
        raise AssertionError('frozen base changed during smoke')
    expected = capture_training(original,progress,identity,permutation,'smoke',config)
    expected_adapter = expected.adapter
    expected_optimizer,expected_rng = _cpu(expected.optimizer),_cpu(expected.rng)
    expected_scheduler,expected_scaler = expected.scheduler,expected.scaler
    expected_progress = progress.as_dict()
    parameter_names = optimizer_parameter_names(original)
    diagnostic['optimizer_parameter_names'] = parameter_names
    frozen_count = sum(p.numel() for p in original.bundle.model.parameters() if not p.requires_grad)
    del expected,original
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()

    def verify_final(objects,progress,branch):
        actual = capture_training(objects,progress,identity,permutation,'smoke',config)
        _compare(expected_adapter,actual.adapter,f'{branch}/adapter')
        # Replace positional IDs in diagnostics with the associated parameter name.
        def named_optimizer(state):
            names = {key:name for group,group_names in zip(state['param_groups'],parameter_names)
                     for key,name in zip(group['params'],group_names)}
            return {'param_groups':state['param_groups'],
                    'state':{names[key]:value for key,value in state['state'].items()}}
        _compare(named_optimizer(expected_optimizer),named_optimizer(actual.optimizer),f'{branch}/optimizer')
        _compare(expected_scheduler,actual.scheduler,f'{branch}/scheduler')
        _compare(expected_scaler,actual.scaler,f'{branch}/scaler')
        _compare(expected_rng,actual.rng,f'{branch}/rng',exact=True)
        if progress.as_dict() != expected_progress or _frozen_hash(objects.bundle.model) != frozen_before:
            raise AssertionError(f'{branch}: progress or frozen base differs')

    # A second independent continuous run separates runtime reproducibility from
    # checkpoint restoration in the very same process/allocation and precision.
    repeated = factory()
    if _weights_hash(adapter_state(repeated.bundle.model)) != initial_hash:
        raise AssertionError('repeat factory did not reconstruct the same fresh adapter')
    repeat_progress = TrainingProgress(permutation_hash=canonical_hash(permutation))
    for index,window in enumerate(windows):
        repeated_result = run_window(repeated.bundle,repeated.optimizer,repeated.scheduler,repeated.scaler,
                                     window,repeat_progress,before_step=observer('continuous_repeat',index))
        record_result('continuous_repeat',repeated_result)
    verify_final(repeated,repeat_progress,'continuous_repeat')
    del repeated
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    resumed = factory()
    resumed_hash = _weights_hash(adapter_state(resumed.bundle.model))
    if resumed_hash != initial_hash:
        raise AssertionError('smoke factory did not reconstruct the same fresh adapter')
    state = load_checkpoint(checkpoint,identity)
    progress = restore_training(state,resumed)
    restored = capture_training(resumed,progress,identity,permutation,'smoke',config)
    for field in ['adapter','optimizer','scheduler','scaler','rng']:
        _compare(getattr(state,field),getattr(restored,field),f'immediate_restore/{field}',exact=True)
    diagnostic['immediate_restore_exact'] = True
    del restored
    for index,window in enumerate(windows[1:],1):
        resumed_result = run_window(resumed.bundle,resumed.optimizer,resumed.scheduler,resumed.scaler,
                                    window,progress,before_step=observer('resumed',index))
        record_result('resumed',resumed_result)
    verify_final(resumed,progress,'resumed')
    result = {'matched':True,'window_sizes':[16,16,12],'global_update':progress.global_update,
              'global_micro_step':progress.global_micro_step,'initial_adapter_hash':initial_hash,
              'resumed_initial_adapter_hash':resumed_hash,'checkpoint':str(checkpoint),
              'gradient_evidence':results[-1].gradient_evidence,'frozen_parameters':frozen_count,
              'continuous_repeat_matched':True,'immediate_restore_exact':True,
              'pre_step_gradients_matched':True,'diagnostic':str(diagnostic_path),
              'rtol':1e-4,'atol':1e-5,'window_metrics':[asdict(r) for r in results]}
    del resumed,state
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return result


def run_smoke(config,audit,reports,run_id):
    require_gpu_allocation(os.environ,socket.gethostname())
    if not run_id.startswith('smoke-'):
        raise ValueError('smoke run_id must use the smoke- namespace')
    from .artifacts import create_run,set_status,start_attempt
    from .preflight import current_input_identity,validate_preflight
    from .model import build_training_model
    from .processor import load_processor,prepare_example,prepare_features
    from .augmentation import augment
    from .data import read_pcm16
    from .paths import audio_path
    gpu = next((r for r in reports if r.stage=='gpu'),None)
    if gpu is None or gpu.runtime is None:
        raise ValueError('GPU preflight runtime required before smoke')
    identity,rows,snapshot,noise,environment = current_input_identity(config,gpu.runtime)
    validate_preflight(identity,reports)
    if audit is None or not audit.valid or audit.source_sha256 != identity.dataset_hash or audit.manifest_sha256 != identity.manifest_hash:
        raise ValueError('smoke requires matching successful official dataset audit')
    run = create_run(config,identity,run_id)
    record = read_json(run/'run-manifest.json'); record.update(mode='smoke',runtime=gpu.runtime.as_dict())
    write_json_atomic(run/'run-manifest.json',record)
    atomic_bytes(run/'manifests/utterances.jsonl',audit.manifest_path.read_bytes())
    write_json_atomic(run/'environment.json',environment)
    for report in reports:
        write_json_atomic(run/'preflight'/f'{report.stage}.json',report.as_dict())
    attempt = start_attempt(run)
    processor = load_processor(snapshot)
    order = epoch_order(len(rows),0)
    selected = [rows[index] for index in order[:44]]
    checks,metadata = [],{}
    def factory():
        bundle = build_training_model(config,snapshot,gpu.runtime)
        optimizer = create_optimizer(bundle,config)
        scheduler = create_scheduler(optimizer,compute_budget(len(rows)))
        scaler = torch.amp.GradScaler('cuda') if gpu.runtime.scaler_enabled else None
        return TrainingObjects(bundle,optimizer,scheduler,scaler)
    try:
        prepared = []
        for row in selected:
            audio,info = read_pcm16(audio_path(config.paths.train_root,row.audio))
            if info['wav_sha256'] != row.wav_sha256: raise ValueError('smoke audio identity changed')
            prepared.append(prepare_example(row,augment(audio,row,0,noise).audio,processor))
        result = compare_resume(factory,prepared,identity,config,run,order,attempt)
        metadata['comparison'] = result
        checks.append({'name':'training_windows','passed':result['window_sizes']==[16,16,12] and result['global_update']==3})
        checks.append({'name':'checkpoint_resume','passed':result['matched']})
        gradients = result['gradient_evidence']
        checks.append({'name':'gradient_connectivity','passed':
             {'encoder_lora_A','decoder_lora_A'}.issubset(gradients.get('nonzero_groups',[]))
             and gradients.get('frozen_gradients')==0 and result['frozen_parameters']>0})
        maximum = factory()
        features = prepare_features(np.zeros(480000,dtype=np.float32),processor.feature_extractor)
        example = PreparedExample(**features,labels=torch.ones(1,447,dtype=torch.long),utterance_id='technical/max-shape')
        probe = run_window(maximum.bundle,maximum.optimizer,maximum.scheduler,maximum.scaler,[example],TrainingProgress())
        metadata['maximum_shape'] = {'feature_shape':[1,80,3000],'label_tokens':447,'metrics':asdict(probe),'official_updates':False}
        checks.append({'name':'maximum_shape','passed':True})
        del maximum
    except Exception as exc:
        checks.append({'name':'execution_error','passed':False,'error':f'{type(exc).__name__}: {exc}'})
    passed = all(c['passed'] for c in checks) and {'training_windows','checkpoint_resume','gradient_connectivity','maximum_shape'}.issubset({c['name'] for c in checks})
    report = SmokeReport(passed,identity,gpu.runtime,[r.utterance_id for r in selected],checks,
                         execution='slurm_gpu',metadata=metadata)
    write_json_atomic(run/'smoke-report.json',report.as_dict())
    set_status(run,'preflight_passed' if passed else 'failed',smoke_passed=passed)
    return report
