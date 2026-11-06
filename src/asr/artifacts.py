"""Run ownership and explicit lifecycle state transitions."""
from datetime import datetime,timezone
import os
import re
import uuid

from .paths import project_path,read_json,write_json_atomic

TRANSITIONS = {
    'created':{'preflight_failed','preflight_passed','running','failed'},
    'preflight_failed':{'preflight_passed','failed'},
    'preflight_passed':{'running','failed'},
    'running':{'interrupted','training_complete','failed'},
    'interrupted':{'running','failed'},
    'failed':{'running','failed'},
    'training_complete':{'export_complete','failed'},
    'export_complete':{'transcripts_complete','failed'},
    'transcripts_complete':set(),
}


def create_run(config,identity,run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',run_id):
        raise ValueError('unsafe run_id')
    if identity.method_hash != config.method_hash:
        raise ValueError('run method identity mismatch')
    run = project_path(config,config.paths.project_root/'runs'/run_id)
    run.mkdir(parents=True,exist_ok=False)
    write_json_atomic(run/'resolved-config.json',config.resolved_dict())
    write_json_atomic(run/'run-manifest.json',{'schema_version':1,'run_id':run_id,'identity':identity.as_dict(),
                'status':'created','created_at':datetime.now(timezone.utc).isoformat()})
    return run


def start_attempt(run_dir):
    attempt = uuid.uuid4().hex
    directory = run_dir/'attempts'/attempt
    directory.mkdir(parents=True,exist_ok=False)
    write_json_atomic(directory/'attempt.json',{'attempt_id':attempt,'started_at':datetime.now(timezone.utc).isoformat(),
                                               'slurm_job_id':os.environ.get('SLURM_JOB_ID')})
    return attempt


def set_status(run_dir,status,**details):
    path = run_dir/'run-manifest.json'
    record = read_json(path)
    previous = record['status']
    if status not in TRANSITIONS.get(previous,set()):
        raise ValueError(f'invalid run transition: {previous} -> {status}')
    record.update(details)
    record.update(status=status,updated_at=datetime.now(timezone.utc).isoformat())
    write_json_atomic(path,record)
    return record
