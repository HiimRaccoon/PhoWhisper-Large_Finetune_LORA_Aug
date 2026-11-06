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
