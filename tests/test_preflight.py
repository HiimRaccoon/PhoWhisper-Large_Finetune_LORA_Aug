from dataclasses import replace
import pytest


def test_missing_assets_fail_preflight_with_report_instead_of_silent_fallback(local_config,tmp_path):
    from asr.preflight import run_preflight
    from asr.paths import read_json
    output=tmp_path/'preflight.json'
    report=run_preflight(local_config,'cpu',output)
    assert report.passed is False
    assert read_json(output)['passed'] is False
    assert {'dataset','snapshot','noise'}.issubset({c['name'] for c in report.checks if not c['passed']})


def test_stale_identity_missing_reports_and_fake_gpu_evidence_are_rejected():
    from test_checkpoint import identity_fixture
    from asr.contracts import PreflightReport,SmokeReport,RuntimeProfile
    from asr.preflight import validate_evidence
    expected=identity_fixture(); runtime=RuntimeProfile('cuda','bfloat16',False,{'metrics':{'targets':192,'trainable_parameters':3932160}})
    checks=[{'name':n,'passed':True} for n in ['dataset','snapshot','noise','dependencies','labels']]
    cpu=PreflightReport('cpu',True,replace(expected,runtime_hash=''),checks,execution='cpu')
    gpu=PreflightReport('gpu',True,expected,checks+[{'name':'allocation','passed':True},{'name':'gpu_probe','passed':True}],runtime,execution='slurm_gpu')
    smoke_checks=[{'name':n,'passed':True} for n in ['training_windows','checkpoint_resume','gradient_connectivity','maximum_shape']]
    smoke=SmokeReport(True,expected,runtime,['s/current'],smoke_checks,execution='slurm_gpu')
    validate_evidence(expected,[cpu,gpu],smoke)
    with pytest.raises(ValueError): validate_evidence(expected,[cpu],smoke)
    with pytest.raises(ValueError,match='identity'):
        validate_evidence(expected,[cpu,gpu],replace(smoke,identity=replace(expected,manifest_hash='stale')))
    with pytest.raises(ValueError): validate_evidence(expected,[cpu,replace(gpu,execution='fixture')],smoke)
def test_environment_manifest_covers_transitive_dependencies():
    from asr.preflight import environment_metadata
    metadata=environment_metadata()
    assert metadata['all_distributions']['safetensors']
    assert metadata['all_distributions']['tokenizers']
    assert metadata['all_distributions']['torch'].startswith('2.9.0')


def runtime_fixture(driver='570.00',uuid='MIG-first'):
    from asr.contracts import RuntimeProfile
    return RuntimeProfile('cuda','bfloat16',False,{'metrics':{'allocation':{
         'driver':driver,'gpu_family':'H200','device_name':'NVIDIA H200 MIG 2g.35gb',
         'profile':'2g.35gb','total_bytes':35*1024**3,'uuid':uuid},'targets':192,'trainable_parameters':3932160}})


def test_runtime_fingerprint_binds_driver_family_profile_but_not_allocation_uuid():
    import copy
    from asr.preflight import runtime_identity
    baseline=runtime_fixture()
    assert runtime_identity(baseline)!=runtime_identity(runtime_fixture('580.00'))
    assert runtime_identity(baseline)==runtime_identity(runtime_fixture(uuid='MIG-next-allocation'))
    for key,value in [('gpu_family','H100'),('profile','1g.18gb'),('device_name','NVIDIA H100')]:
        changed=copy.deepcopy(baseline)
        changed.probe['metrics']['allocation'][key]=value
        assert runtime_identity(baseline)!=runtime_identity(changed)


def test_current_runtime_fingerprint_inspects_live_hardware_instead_of_old_report(monkeypatch):
    from asr.preflight import current_runtime_identity,runtime_identity
    from asr import precision
    previous=runtime_fixture(); live=runtime_fixture('580.00')
    monkeypatch.setattr(precision,'inspect_allocated_gpu',lambda:live.probe['metrics']['allocation'])
    assert current_runtime_identity(previous)==runtime_identity(live)
    assert current_runtime_identity(previous)!=runtime_identity(previous)
