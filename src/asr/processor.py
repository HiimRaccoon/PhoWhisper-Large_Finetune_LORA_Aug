"""Fixed-window features and non-truncating labels for the local Whisper snapshot."""
import unicodedata

import numpy as np
import torch

from .contracts import PreparedExample


def load_processor(snapshot):
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(str(snapshot.path), local_files_only=True, trust_remote_code=False)
    if processor.feature_extractor.sampling_rate != 16000 or processor.feature_extractor.feature_size != 80:
        raise ValueError('processor must use 16 kHz and 80 mel bins')
    processor.tokenizer.set_prefix_tokens(language='vi',task='transcribe',predict_timestamps=False)
    return processor


def prepare_features(audio: np.ndarray, feature_extractor) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not 0 < len(audio) <= 480000 or not np.isfinite(audio).all():
        raise ValueError('audio view must be finite mono, >0 and <=480000 samples')
    result = feature_extractor(audio, sampling_rate=16000, padding='max_length', max_length=480000,
                               truncation=False, return_attention_mask=True, return_tensors='pt')
    features,mask = result['input_features'],result['attention_mask'].long()
    if tuple(features.shape) != (1,80,3000) or tuple(mask.shape) != (1,3000):
        raise ValueError('Whisper feature/mask shape mismatch')
    expected = (torch.arange(3000) * 160 < len(audio)).long().unsqueeze(0)
    if not torch.equal(mask.cpu(),expected) or not torch.isfinite(features).all():
        raise ValueError('Whisper frame mask or finite-feature contract mismatch')
    return {'input_features':features.float(),'attention_mask':mask}


def tokenize_labels(row, tokenizer):
    reference = unicodedata.normalize('NFC',row.reference).strip()
    if not reference:
        raise ValueError(f'{row.utterance_id}: empty transcript')
    tokenizer.set_prefix_tokens(language='vi',task='transcribe',predict_timestamps=False)
    ids = list(tokenizer(reference, truncation=False)['input_ids'])
    start = tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    if ids and ids[0] == start:
        ids = ids[1:]
    if not ids or len(ids) > 447:
        raise ValueError(f'{row.utterance_id}: labels exceed 447 tokens or are empty')
    if ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f'{row.utterance_id}: missing label EOS')
    return torch.tensor([ids],dtype=torch.long)


def prepare_example(row, audio: np.ndarray, processor) -> PreparedExample:
    labels = tokenize_labels(row,processor.tokenizer)
    features = prepare_features(audio,processor.feature_extractor)
    return PreparedExample(**features, labels=labels,utterance_id=row.utterance_id)


def collate_examples(examples: list[PreparedExample]) -> dict:
    if not examples:
        raise ValueError('cannot collate an empty batch')
    labels = torch.nn.utils.rnn.pad_sequence([e.labels.squeeze(0) for e in examples],
                                           batch_first=True,padding_value=-100)
    return {'input_features':torch.cat([e.input_features for e in examples]),
            'attention_mask':torch.cat([e.attention_mask for e in examples]),'labels':labels}
