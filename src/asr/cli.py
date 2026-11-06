"""Operator CLI; official GPU execution is unreachable without matching admission evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import load_config
from .paths import atomic_bytes,project_path,read_json,sha256_file,write_json_atomic


def _common(parser):
    parser.add_argument('--config',type=Path,default=Path('configs/official.yaml'))
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--noise-manifest',type=Path)


def _parser():
    parser=argparse.ArgumentParser(prog='asr',description='Locked PhoWhisper LoRA ASR lifecycle')
    commands=parser.add_subparsers(dest='command',required=True)
    audit=commands.add_parser('audit'); _common(audit)
    audit.add_argument('--output',type=Path,default=Path('manifests/train'))
    preflight=commands.add_parser('preflight'); _common(preflight)
    preflight.add_argument('--stage',choices=['cpu','gpu'],required=True)
    preflight.add_argument('--output',type=Path,required=True)
    smoke=commands.add_parser('smoke'); _common(smoke)
    smoke.add_argument('--run-id',required=True)
    smoke.add_argument('--cpu-report',type=Path,required=True)
    smoke.add_argument('--gpu-report',type=Path,required=True)
    for name in ['train','resume']:
        run=commands.add_parser(name); _common(run)
        run.add_argument('--run-id',required=True)
        for flag in ['cpu-report','gpu-report','smoke-report']:
            run.add_argument('--'+flag,type=Path,required=name=='train')
        if name=='resume':
            run.add_argument('--checkpoint',type=Path,required=True)
    return parser


def _config(args):
    config=load_config(args.config)
    updates={name:getattr(args,name) for name in ['data_root','snapshot','noise_manifest'] if getattr(args,name,None) is not None}
    return config.with_paths(**updates) if updates else config


def _run_command(args,config):
    from .artifacts import create_run
    from .contracts import PreflightReport,SmokeReport
    from .preflight import check_admission,current_input_identity
    from .training import execute_run,compute_budget
    if args.command=='resume':
        # The name is validated before joining paths, even on resume.
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',args.run_id):
            raise ValueError('unsafe run_id')
        run=project_path(config,config.paths.project_root/'runs'/args.run_id)
    else:
        run=None
        if args.run_id.startswith('smoke-'):
            raise ValueError('smoke- run IDs are reserved for smoke tests')
    files={name:getattr(args,name+'_report') or run/'preflight'/f'{name}.json' for name in ['cpu','gpu','smoke']}
    cpu=PreflightReport.from_dict(read_json(project_path(config,files['cpu'])))
    gpu=PreflightReport.from_dict(read_json(project_path(config,files['gpu'])))
    smoke=SmokeReport.from_dict(read_json(project_path(config,files['smoke'])))
    identity=check_admission(config,[cpu,gpu],smoke)
    if run is None:
        run=create_run(config,identity,args.run_id)
        current,rows,snapshot,noise,environment=current_input_identity(config,gpu.runtime,verify_audio=False)
        for name in ['utterances.jsonl','audit.json','duplicates.json','conflicts.json']:
            atomic_bytes(run/'manifests'/name,(config.paths.manifest_dir/name).read_bytes())
        write_json_atomic(run/'manifests/snapshot.json',asdict(snapshot))
        write_json_atomic(run/'manifests/noise.json',asdict(noise))
        atomic_bytes(run/'manifests/noise-manifest.json',noise.path.read_bytes())
        write_json_atomic(run/'environment.json',environment)
        record=read_json(run/'run-manifest.json')
        record.update(runtime=gpu.runtime.as_dict(),budget=asdict(compute_budget(len(rows))),
                      model_revision=config.model.revision,lora_scope=config.lora.scope,seed=42)
        write_json_atomic(run/'run-manifest.json',record)
    for name,report in [('cpu',cpu),('gpu',gpu),('smoke',smoke)]:
        write_json_atomic(run/'preflight'/f'{name}.json',report.as_dict())
    progress=execute_run(config,run,getattr(args,'checkpoint',None))
    print(json.dumps({'run_id':run.name,'completed_epochs':progress.completed_epochs,'updates':progress.global_update}))
    return 0 if progress.completed_epochs==10 else 3


def main(argv=None):
    args=_parser().parse_args(argv)
    try:
        config=_config(args)
        if args.command=='audit':
            from .audit import audit_train
            result=audit_train(config,args.output)
            print(json.dumps({'valid':result.valid,'sources':result.source_count,'utterances':result.utterance_count,
                              'report':str(result.report_path)}))
            return 0 if result.valid else 1
        if args.command=='preflight':
            from .preflight import run_preflight
            result=run_preflight(config,args.stage,args.output)
            print(json.dumps({'passed':result.passed,'stage':args.stage,'report':str(project_path(config,args.output))}))
            return 0 if result.passed else 1
        if args.command in {'train','resume'}:
            return _run_command(args,config)
        if args.command=='smoke':
            from .audit import audit_train
            from .contracts import PreflightReport
            from .smoke import run_smoke
            reports=[PreflightReport.from_dict(read_json(project_path(config,path)))
                     for path in [args.cpu_report,args.gpu_report]]
            audit=audit_train(config,config.paths.manifest_dir)
            report=run_smoke(config,audit,reports,args.run_id)
            print(json.dumps({'passed':report.passed,'report':str(config.paths.project_root/'runs'/args.run_id/'smoke-report.json')}))
            return 0 if report.passed else 1
        raise ValueError('unsupported lifecycle command')
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
