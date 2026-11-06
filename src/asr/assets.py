"""Validate local asset provenance and hashes; runtime never downloads assets."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import canonical_hash
from .contracts import NoiseEntry, NoiseManifest, SnapshotIdentity
from .data import read_pcm16
from .paths import project_path, read_json, resolve_beneath, sha256_file


def inventory_files(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if '.cache' in relative.parts or relative.as_posix() == '.snapshot-origin.json':
            continue
        safe = resolve_beneath(root, relative.as_posix())
        if safe.is_file():
            result[relative.as_posix()] = sha256_file(safe)
    return result


def prepare_snapshot(config,download=None) -> SnapshotIdentity:
    """Explicit online preparation, never called by training, export or inference."""
    import os,uuid
    from filelock import FileLock
    from .paths import write_json_atomic
    root=project_path(config,config.paths.snapshot)
    if root.exists(): raise FileExistsError(root)
    root.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root.parent/'.snapshot-prepare.lock'),timeout=0):
        if root.exists(): raise FileExistsError(root)
        temporary=root.parent/f'.tmp-snapshot-{uuid.uuid4().hex}'
        if download is None:
            from huggingface_hub import snapshot_download
            download=snapshot_download
        download(repo_id=config.model.base,revision=config.model.revision,local_dir=str(temporary))
        write_json_atomic(temporary/'.snapshot-origin.json',{'repository':config.model.base,
                          'revision':config.model.revision,'files':inventory_files(temporary)})
        verify_snapshot(config.with_paths(snapshot=temporary))
        os.rename(temporary,root)
    return verify_snapshot(config)


def verify_snapshot(config) -> SnapshotIdentity:
    root = project_path(config, config.paths.snapshot)
    origin = read_json(root / '.snapshot-origin.json')
    if origin.get('repository') != config.model.base or origin.get('revision') != config.model.revision:
        raise ValueError('snapshot repository/revision mismatch')
    files = inventory_files(root)
    if files != origin.get('files'):
        raise ValueError('snapshot file inventory/hash mismatch')
    required = {'config.json','preprocessor_config.json','tokenizer_config.json','generation_config.json'}
    if not required.issubset(files):
        raise ValueError('snapshot missing required config/processor files')
    if 'tokenizer.json' not in files and not {'vocab.json','merges.txt'}.issubset(files):
        raise ValueError('snapshot tokenizer is incomplete')
    weights = [name for name in files if name.endswith(('.safetensors','.bin'))]
    if not weights:
        raise ValueError('snapshot missing model weights')
    for index_name in ('model.safetensors.index.json','pytorch_model.bin.index.json'):
        if index_name in files:
            mapping = read_json(root/index_name).get('weight_map', {})
            if not mapping or not set(mapping.values()).issubset(files):
                raise ValueError('snapshot weight shards are incomplete')
    architecture = read_json(root/'config.json')
    expected = {'model_type':'whisper','encoder_layers':32,'decoder_layers':32,'d_model':1280,'num_mel_bins':80}
    if any(architecture.get(key) != value for key,value in expected.items()):
        raise ValueError('snapshot architecture mismatch')
    identity = {'repository':config.model.base,'revision':config.model.revision,'files':files}
    return SnapshotIdentity(root, config.model.base, config.model.revision, files, canonical_hash(identity))


def verify_noise(config) -> NoiseManifest:
    manifest = project_path(config, config.paths.noise_manifest)
    raw = read_json(manifest)
    if not isinstance(raw,list) or not raw:
        raise ValueError('noise manifest must be a nonempty ordered array')
    entries = []
    for index,entry in enumerate(raw):
        if not isinstance(entry,dict) or entry.get('reviewed') is not True or entry.get('contains_speech') is not False:
            raise ValueError(f'noise entry {index} requires review and non-speech evidence')
        license = entry.get('license')
        provenance = entry.get('source_url') or entry.get('license_url') or entry.get('licensing_evidence')
        if not isinstance(license,str) or not license.strip() or not isinstance(provenance,str) or not provenance.strip():
            raise ValueError(f'noise entry {index} requires license and provenance')
        path = resolve_beneath(manifest.parent, entry.get('path',''))
        audio,metadata = read_pcm16(path, max_seconds=None)
        if metadata['wav_sha256'] != entry.get('sha256'):
            raise ValueError(f'noise entry {index} hash mismatch')
        if np.mean(np.square(audio.astype(np.float64))) <= 0:
            raise ValueError(f'noise entry {index} has zero power')
        entries.append(NoiseEntry(path, metadata['wav_sha256'], metadata['pcm_sha256'], license,
                                  provenance, metadata['num_frames']))
    return NoiseManifest(manifest, sha256_file(manifest), tuple(entries))


@lru_cache(maxsize=8)
def load_noise(entry: NoiseEntry) -> np.ndarray:
    audio,metadata = read_pcm16(entry.path, max_seconds=None)
    if metadata['wav_sha256'] != entry.sha256:
        raise ValueError('noise asset changed since verification')
    audio.flags.writeable = False
    return audio
