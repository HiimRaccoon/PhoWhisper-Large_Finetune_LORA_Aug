"""Portable fake scheduler checks; actual signal/filesystem tests are marked linux."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
BASH=shutil.which('bash') if sys.platform!='win32' else 'C:/Program Files/Git/bin/bash.exe'


@pytest.fixture
def scheduler(tmp_path):
    root=tmp_path/'project with spaces'; root.mkdir()
    (root/'configs').mkdir(); (root/'configs/official.yaml').write_text('fixture')
    (root/'.venv/bin').mkdir(parents=True)
    fake=tmp_path/'fakebin'; fake.mkdir()
    capture=tmp_path/'capture.py'
    capture.write_text('import os,sys,json,pathlib\npathlib.Path(os.environ["ASR_TEST_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\nsys.exit(int(os.environ.get("ASR_TEST_EXIT","0")))\n')
    launcher='#!/usr/bin/env bash\nexec "$ASR_TEST_PYTHON" "$ASR_TEST_CAPTURE_SCRIPT" "$@"\n'
    for name in ['sbatch','srun']:
        path=fake/name; path.write_text(launcher,encoding='utf-8',newline='\n'); path.chmod(0o755)
    python=root/'.venv/bin/python'; python.write_text(launcher,encoding='utf-8',newline='\n'); python.chmod(0o755)
    env=dict(os.environ,ASR_PROJECT_ROOT=root.as_posix(),ASR_ACCOUNT='account',ASR_PARTITION='partition',
             ASR_GRES='gpu:profile:1',ASR_CPUS='4',ASR_MEM='48G',ASR_TIME='01:00:00',ASR_SIGNAL_SECONDS='120',
             ASR_DATA_ROOT=(root.parent/'read only data').as_posix(),
             ASR_TEST_PYTHON=Path(sys.executable).as_posix(),ASR_TEST_CAPTURE_SCRIPT=capture.as_posix(),
             ASR_TEST_CAPTURE=(tmp_path/'captured.json').as_posix(),PATH=str(fake)+os.pathsep+os.environ['PATH'])
    env.pop('ASR_SLURM_ENV',None)
    return root,env,tmp_path/'captured.json'


def bash(script,args,env):
    return subprocess.run([BASH,(ROOT/'scripts'/script).as_posix(),*args],env=env,capture_output=True,text=True)


@pytest.mark.parametrize('missing',['ASR_ACCOUNT','ASR_PARTITION','ASR_GRES'])
def test_submit_rejects_missing_resources(scheduler,missing):
    root,env,captured=scheduler; env.pop(missing)
    result=bash('submit_job.sh',['train','--run-id','official'],env)
    assert result.returncode!=0 and missing in result.stderr and not captured.exists()


def test_submit_preserves_arguments_and_creates_log_parents(scheduler):
    root,env,captured=scheduler
    result=bash('submit_job.sh',['resume','--run-id','official','--checkpoint','path with spaces'],env)
    assert result.returncode==0,result.stderr
    args=json.loads(captured.read_text())
    assert '--ntasks=1' in args and '--gres=gpu:profile:1' in args
    assert '--signal=B:USR1@120' in args
    assert args[-5:]==['resume','--run-id','official','--checkpoint','path with spaces']
    assert any('train.slurm' in arg for arg in args)
    assert (root/'runs/slurm').is_dir()
    env['ASR_TEST_EXIT']='23'
    assert bash('submit_job.sh',['infer','--model','a'],env).returncode==23


def test_step_preserves_exit_status_and_restarts_same_job(scheduler):
    root,env,captured=scheduler
    env['SLURM_JOB_ID']='same-id'; env['ASR_TEST_EXIT']='7'
    for _ in range(2):
        result=bash('train.slurm',['resume','--run-id','official','--checkpoint','with space'],env)
        assert result.returncode==7,result.stderr
        args=json.loads(captured.read_text())
        assert '--ntasks=1' in args and args[args.index('asr')+1]=='resume'
        assert args[-4:]==['--run-id','official','--checkpoint','with space']


def test_runtime_rejects_missing_allocation(scheduler):
    root,env,captured=scheduler; env.pop('SLURM_JOB_ID',None)
    result=bash('infer.slurm',['--model','a'],env)
    assert result.returncode!=0 and 'SLURM_JOB_ID' in result.stderr and not captured.exists()


def test_bash_syntax_and_no_interpolated_sbatch_resources():
    scripts=list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'scripts').glob('*.slurm'))
    assert scripts
    for path in scripts:
        result=subprocess.run([BASH,'-n',path.as_posix()],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert not any(line.startswith('#SBATCH') and '$' in line for line in path.read_text().splitlines())


@pytest.mark.linux
def test_posix_warning_signal_reaches_step_and_preserves_exit_status(scheduler):
    import signal,time
    assert sys.platform.startswith('linux'),'G1.linux requires actual Linux'
    root,env,captured=scheduler
    fake=Path(env['PATH'].split(os.pathsep)[0])
    (fake/'srun').write_text('#!/usr/bin/env bash\nwhile [[ $1 == --* ]]; do shift; done\nexec "$@"\n')
    (fake/'srun').chmod(0o755)
    worker=root/'worker.py'
    worker.write_text('import os,signal,time,pathlib\np=pathlib.Path(os.environ["ASR_TEST_CAPTURE"])\n'
        'def stop(*_):\n p.write_text("USR1 received")\n raise SystemExit(3)\n'
        'signal.signal(signal.SIGUSR1,stop)\np.write_text("ready")\nwhile True: time.sleep(.02)\n')
    launcher=root/'.venv/bin/python'
    launcher.write_text('#!/usr/bin/env bash\nexec "$ASR_TEST_PYTHON" "$ASR_TEST_WORKER"\n'); launcher.chmod(0o755)
    env.update(SLURM_JOB_ID='same-id',ASR_TEST_WORKER=worker.as_posix())
    process=subprocess.Popen([BASH,(ROOT/'scripts/train.slurm').as_posix(),'train','--run-id','fixture'],env=env)
    try:
        deadline=time.monotonic()+15
        while not captured.exists() and time.monotonic()<deadline: time.sleep(.02)
        assert captured.read_text()=='ready'
        os.kill(process.pid,signal.SIGUSR1)
        assert process.wait(timeout=10)==3
        assert captured.read_text()=='USR1 received'
    finally:
        if process.poll() is None: process.kill(); process.wait()
