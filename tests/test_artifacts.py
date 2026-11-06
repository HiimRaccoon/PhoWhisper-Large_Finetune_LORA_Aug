from dataclasses import replace
import pytest


def test_run_ids_attempts_and_transitions_preserve_existing_artifacts(local_config):
    from asr.artifacts import create_run,start_attempt,set_status
    from asr.paths import read_json
    from test_checkpoint import identity_fixture
    identity=replace(identity_fixture(),method_hash=local_config.method_hash)
    run=create_run(local_config,identity,'run1')
    with pytest.raises(FileExistsError): create_run(local_config,identity,'run1')
    first=start_attempt(run); second=start_attempt(run)
    assert first!=second and (run/'attempts'/first).is_dir()
    set_status(run,'running')
    with pytest.raises(ValueError,match='transition'): set_status(run,'transcripts_complete')
    set_status(run,'interrupted'); set_status(run,'running')
    assert read_json(run/'run-manifest.json')['status']=='running'
    with pytest.raises(ValueError): create_run(local_config,identity,'../escape')


def test_new_attempt_recovers_running_state_only_under_exclusive_session_lock(local_config):
    from filelock import FileLock
    from asr.artifacts import create_run,start_attempt,set_status,begin_attempt
    from asr.paths import read_json
    from test_checkpoint import identity_fixture
    identity=replace(identity_fixture(),method_hash=local_config.method_hash)
    run=create_run(local_config,identity,'crashed')
    old=start_attempt(run); set_status(run,'running',latest_attempt=old)
    # A hard kill cannot update status; OS ownership is the recovery authority.
    with FileLock(str(run/'.session.lock'),timeout=0) as owner:
        new=begin_attempt(run,owner)
    record=read_json(run/'run-manifest.json')
    assert new!=old and record['status']=='running' and record['latest_attempt']==new
    assert record['recovered_attempt']==old
    assert read_json(run/'attempts'/new/'attempt.json')['recovered_attempt']==old
    assert (run/'attempts'/old/'attempt.json').exists()
    with pytest.raises(RuntimeError,match='lock'): begin_attempt(run,owner)
