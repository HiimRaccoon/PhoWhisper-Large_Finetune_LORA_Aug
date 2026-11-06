"""Read-only Train audit; any rejected row or identity mismatch blocks admission."""
from collections import defaultdict
from dataclasses import replace
import unicodedata

from .config import TRAIN_SHA256
from .contracts import AuditResult
from .data import project_sources, read_pcm16
from .paths import audio_path, project_path, sha256_file, write_json_atomic, write_jsonl_atomic


def audit_train(config, output_dir) -> AuditResult:
    output = project_path(config, output_dir)
    source = config.paths.train_root / 'train.jsonl'
    errors, audited, groups = [], [], defaultdict(list)
    source_hash = sha256_file(source) if source.is_file() else ''
    try:
        rows = project_sources(source)
    except (OSError, ValueError) as exc:
        errors.append({'kind':'source_error','reason':str(exc)})
        rows = []
    count = len({r.sample_id for r in rows})
    if source_hash != TRAIN_SHA256:
        errors.append({'kind':'dataset_identity','reason':'Train JSONL SHA-256 mismatch'})
    if count != config.data.source_samples or len(rows) != config.data.expected_utterances:
        errors.append({'kind':'dataset_count','reason':f'sources={count}, utterances={len(rows)}'})
    for row in rows:
        try:
            reference = unicodedata.normalize('NFC', row.reference).strip()
            if not reference:
                raise ValueError('empty transcript after NFC + trim')
            _, metadata = read_pcm16(audio_path(config.paths.train_root, row.audio))
            updated = replace(row, **metadata, normalized_reference=reference)
            audited.append(updated)
            groups[updated.pcm_sha256].append(updated)
        except (OSError, ValueError) as exc:
            errors.append({'kind':'utterance_error','utterance_id':row.utterance_id,
                           'audio':row.audio,'reason':str(exc)})
    duplicates, conflicts = [], []
    for digest, members in groups.items():
        if len(members) > 1:
            record = {'pcm_sha256':digest,'utterance_ids':[r.utterance_id for r in members],
                      'wav_sha256':[r.wav_sha256 for r in members]}
            duplicates.append(record)
            if len({r.normalized_reference for r in members}) > 1:
                conflicts.append(record)
                errors.append({'kind':'transcript_conflict',**record})
    manifest = output / 'utterances.jsonl'
    write_jsonl_atomic(manifest, (r.as_dict() for r in audited))
    write_json_atomic(output / 'duplicates.json', duplicates)
    write_json_atomic(output / 'conflicts.json', conflicts)
    result = AuditResult(not errors, count, len(rows), manifest, output/'audit.json', source_hash,
                         sha256_file(manifest), errors, len(duplicates), len(conflicts))
    write_json_atomic(result.report_path, result.as_dict())
    return result
