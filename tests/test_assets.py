import hashlib
import json
import pytest


def snapshot_fixture(config):
    root=config.paths.snapshot; root.mkdir(parents=True)
    content={'config.json':json.dumps({'model_type':'whisper','encoder_layers':32,'decoder_layers':32,'d_model':1280,'num_mel_bins':80}),
             'preprocessor_config.json':'{}','tokenizer_config.json':'{}','tokenizer.json':'{}',
             'generation_config.json':'{}','model.safetensors':'fixture weights'}
    for name,data in content.items(): (root/name).write_text(data,encoding='utf-8')
    origin={'repository':config.model.base,'revision':config.model.revision,
            'files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in content}}
    (root/'.snapshot-origin.json').write_text(json.dumps(origin),encoding='utf-8')
    return root


def test_snapshot_identity_requires_exact_origin_and_untampered_files(local_config):
    from asr.assets import verify_snapshot
    root=snapshot_fixture(local_config)
    identity=verify_snapshot(local_config)
    assert identity.revision == local_config.model.revision
    assert len(identity.files)==6
    (root/'model.safetensors').write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError,match='hash'):
        verify_snapshot(local_config)


def test_noise_requires_license_review_hash_and_positive_power(local_config,make_wav):
    from asr.assets import verify_noise
    root=local_config.paths.noise_manifest.parent
    wav=make_wav(root/'background.wav')
    entry={'path':'background.wav','sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
           'license':'CC0-1.0','reviewed':True,'contains_speech':False,'source_url':'https://example.invalid/noise'}
    local_config.paths.noise_manifest.write_text(json.dumps([entry]),encoding='utf-8')
    result=verify_noise(local_config)
    assert len(result.entries)==1 and result.entries[0].num_frames==4
    entry['reviewed']=False
    local_config.paths.noise_manifest.write_text(json.dumps([entry]),encoding='utf-8')
    with pytest.raises(ValueError,match='review'):
        verify_noise(local_config)


def test_noise_assets_can_exceed_thirty_seconds(local_config,make_wav):
    from asr.assets import verify_noise
    root=local_config.paths.noise_manifest.parent
    wav=make_wav(root/'long.wav',samples=[100]*496000)
    entry={'path':'long.wav','sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
           'license':'CC0-1.0','reviewed':True,'contains_speech':False,'source_url':'https://example.invalid/noise'}
    local_config.paths.noise_manifest.write_text(json.dumps([entry]),encoding='utf-8')
    assert verify_noise(local_config).entries[0].num_frames==496000
