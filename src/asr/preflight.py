"""Identity-bound CPU/GPU checks and official-run admission."""
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import socket
import sys

from .assets import verify_snapshot,verify_noise
from .audit import audit_train
from .config import TRAIN_SHA256,canonical_hash
from .contracts import ArtifactIdentity,PreflightReport,UtteranceRow
from .paths import audio_path,project_path,read_json,read_jsonl,sha256_file,write_json_atomic,write_jsonl_atomic

CPU_CHECKS = {'dataset','snapshot','noise','dependencies','labels'}
SMOKE_CHECKS = {'training_windows','checkpoint_resume','gradient_connectivity','maximum_shape'}


def environment_metadata() -> dict:
    expected = {'torch':'2.9.0','transformers':'4.57.1','peft':'0.17.1','accelerate':'1.10.1',
                'numpy':'2.3.4','huggingface-hub':'0.36.0'}
    versions = {name:importlib.metadata.version(name) for name in [*expected,'PyYAML','safetensors','filelock']}
    if sys.version_info[:2] != (3,11) or any(versions[name].split('+')[0] != version for name,version in expected.items()):
        raise ValueError(f'candidate dependency/Python compatibility mismatch: {versions}')
    package = Path(__file__).resolve().parent
    code = {p.name:sha256_file(p) for p in sorted(package.glob('*.py'))}
    distributions={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
    return {'python':sys.version.split()[0],'platform':sys.platform,'versions':versions,
            'all_distributions':dict(sorted(distributions.items())),'source_files':code}


def runtime_identity(runtime,allocation=None) -> str:
    if runtime is None:
        return ''
    import torch
    hardware=None
    if runtime.device=='cuda':
        allocation=allocation if allocation is not None else runtime.probe.get('metrics',{}).get('allocation',{})
        required={'driver','gpu_family','device_name','profile','total_bytes'}
        if not required.issubset(allocation): raise ValueError('verified GPU/driver identity is incomplete')
        hardware={key:allocation[key] for key in required if key!='driver'}
        hardware['driver_versions']=sorted(set(allocation['driver'].splitlines()))
    return canonical_hash({'device':runtime.device,'compute_dtype':runtime.compute_dtype,
          'scaler_enabled':runtime.scaler_enabled,'torch':str(torch.__version__),'cuda':torch.version.cuda,
          'hardware':hardware,
          'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
          'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
          'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark})


def current_runtime_identity(runtime):
    from .precision import inspect_allocated_gpu
    return runtime_identity(runtime,inspect_allocated_gpu() if runtime and runtime.device=='cuda' else None)


def load_audited_rows(config,verify_audio=True):
    root = project_path(config,config.paths.manifest_dir)
    report = read_json(root/'audit.json')
    manifest = root/'utterances.jsonl'
    if report.get('valid') is not True or report.get('source_count') != 1545 or report.get('utterance_count') != 5036:
        raise ValueError('a successful official Train audit is required')
    if report.get('source_sha256') != TRAIN_SHA256 or sha256_file(config.paths.train_root/'train.jsonl') != TRAIN_SHA256:
        raise ValueError('Train source identity changed')
    if sha256_file(manifest) != report.get('manifest_sha256'):
        raise ValueError('utterance manifest hash changed')
    rows = [UtteranceRow(**r) for r in read_jsonl(manifest)]
    if len(rows) != 5036 or len({r.utterance_id for r in rows}) != 5036 or len({r.sample_id for r in rows}) != 1545:
        raise ValueError('utterance manifest coverage mismatch')
    if verify_audio:
        for row in rows:
            if sha256_file(audio_path(config.paths.train_root,row.audio)) != row.wav_sha256:
                raise ValueError(f'audio changed: {row.utterance_id}')
    return rows,report


def current_input_identity(config,runtime,verify_audio=True):
    rows,audit = load_audited_rows(config,verify_audio)
    snapshot,noise = verify_snapshot(config),verify_noise(config)
    environment = environment_metadata()
    identity = ArtifactIdentity(config.method_hash,audit['source_sha256'],audit['manifest_sha256'],
                                  snapshot.identity_hash,noise.sha256,current_runtime_identity(runtime),canonical_hash(environment))
    return identity,rows,snapshot,noise,environment


def run_preflight(config,stage,output):
    if stage not in {'cpu','gpu'}:
        raise ValueError('preflight stage must be cpu or gpu')
    output = project_path(config,output)
    checks,values = [],{}
    def check(name,operation):
        try:
            value = operation()
            values[name] = value
            checks.append({'name':name,'passed':True})
        except Exception as exc:
            checks.append({'name':name,'passed':False,'error':f'{type(exc).__name__}: {exc}'})
    if stage == 'gpu':
        from .precision import require_gpu_allocation
        check('allocation',lambda:require_gpu_allocation(os.environ,socket.gethostname()))
    def dataset():
        result = audit_train(config,config.paths.manifest_dir)
        if not result.valid:
            raise ValueError(f'dataset audit failed: {result.errors[:5]}')
        return result
    check('dataset',dataset)
    check('snapshot',lambda:verify_snapshot(config))
    check('noise',lambda:verify_noise(config))
    check('dependencies',environment_metadata)
    def labels():
        if 'snapshot' not in values or 'dataset' not in values:
            raise ValueError('label checks require valid dataset and snapshot')
        from .processor import load_processor,tokenize_labels
        processor = load_processor(values['snapshot'])
        rows,_ = load_audited_rows(config,verify_audio=False)
        lengths = [{'utterance_id':r.utterance_id,'label_token_count':tokenize_labels(r,processor.tokenizer).shape[-1]} for r in rows]
        write_jsonl_atomic(output.with_name(output.stem+'.labels.jsonl'),lengths)
        return {'utterances':len(lengths),'max_label_tokens':max(r['label_token_count'] for r in lengths)}
    check('labels',labels)
    if stage == 'gpu':
        def gpu_probe():
            if 'allocation' not in values or not CPU_CHECKS.issubset(values):
                raise ValueError('GPU probe requires allocation and all CPU prerequisites')
            from .precision import probe_runtime
            return probe_runtime(config,values['snapshot'])
        check('gpu_probe',gpu_probe)
    passed = all(c['passed'] for c in checks)
    runtime = values.get('gpu_probe')
    identity = None
    if passed:
        audit,snapshot,noise,environment = (values[k] for k in ['dataset','snapshot','noise','dependencies'])
        identity = ArtifactIdentity(config.method_hash,audit.source_sha256,audit.manifest_sha256,
                                   snapshot.identity_hash,noise.sha256,runtime_identity(runtime),canonical_hash(environment))
    report = PreflightReport(stage,passed,identity,checks,runtime,
                  'slurm_gpu' if stage=='gpu' and passed else 'cpu' if stage=='cpu' else 'unverified',
                  {'method_hash':config.method_hash,'environment':values.get('dependencies'),
                   'label_summary':values.get('labels')})
    write_json_atomic(output,report.as_dict())
    return report


def _passed_checks(report,required):
    return report.passed is True and required.issubset({c.get('name') for c in report.checks if c.get('passed') is True}) and all(c.get('passed') is True for c in report.checks)


def validate_preflight(expected,reports):
    if len(reports) != 2 or {r.stage for r in reports} != {'cpu','gpu'}:
        raise ValueError('matching CPU and GPU preflight reports are required')
    cpu = next(r for r in reports if r.stage=='cpu')
    gpu = next(r for r in reports if r.stage=='gpu')
    if not _passed_checks(cpu,CPU_CHECKS) or not _passed_checks(gpu,CPU_CHECKS|{'allocation','gpu_probe'}):
        raise ValueError('preflight checks have not all passed')
    if cpu.identity is None or gpu.identity is None:
        raise ValueError('missing evidence identity')
    shared = expected.as_dict(); shared.pop('runtime_hash')
    cpu_shared = cpu.identity.as_dict(); cpu_shared.pop('runtime_hash')
    if cpu_shared != shared or gpu.identity != expected:
        raise ValueError('preflight input identity mismatch')
    if cpu.execution != 'cpu' or gpu.execution != 'slurm_gpu':
        raise ValueError('actual allocated-GPU evidence is required')
    if not gpu.runtime or gpu.runtime.device != 'cuda' or gpu.runtime.compute_dtype not in {'bfloat16','float16'}:
        raise ValueError('invalid verified GPU runtime')
    if gpu.runtime.scaler_enabled != (gpu.runtime.compute_dtype=='float16'):
        raise ValueError('GPU scaler profile mismatch')
    metrics = gpu.runtime.probe.get('metrics',{})
    if metrics.get('targets') != 192 or metrics.get('trainable_parameters') != 3932160:
        raise ValueError('actual LoRA count evidence missing')
    return gpu


def validate_evidence(expected,reports,smoke):
    gpu = validate_preflight(expected,reports)
    if smoke.identity != expected:
        raise ValueError('smoke input identity mismatch')
    if smoke.execution != 'slurm_gpu':
        raise ValueError('actual allocated-GPU smoke evidence is required')
    if not _passed_checks(smoke,SMOKE_CHECKS) or not smoke.fixture_ids:
        raise ValueError('full-model smoke checks have not passed')
    if smoke.runtime.compute_dtype != gpu.runtime.compute_dtype or smoke.runtime.scaler_enabled != gpu.runtime.scaler_enabled:
        raise ValueError('smoke precision profile mismatch')


def check_admission(config,reports,smoke):
    from .precision import require_gpu_allocation
    import torch
    require_gpu_allocation(os.environ,socket.gethostname())
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('official execution requires one actual allocated GPU')
    gpu = next((r for r in reports if r.stage=='gpu'),None)
    if gpu is None or gpu.runtime is None:
        raise ValueError('GPU preflight runtime missing')
    expected,*_ = current_input_identity(config,gpu.runtime,verify_audio=True)
    validate_evidence(expected,reports,smoke)
    return expected
