from types import SimpleNamespace
import numpy as np
import pytest


class TokenizerFixture:
    eos_token_id=0
    pad_token_id=0

    def set_prefix_tokens(self,**kwargs):
        self.prefix=kwargs

    def convert_tokens_to_ids(self,token):
        return 1

    def __call__(self,text,**kwargs):
        self.last_text=text
        count=int(text[2:]) if text.startswith('n=') else 1
        return {'input_ids':[1,2,3,4]+[5]*count+[0]}


def processor_fixture():
    from transformers import WhisperFeatureExtractor
    return SimpleNamespace(feature_extractor=WhisperFeatureExtractor(feature_size=80,sampling_rate=16000),tokenizer=TokenizerFixture())


@pytest.mark.parametrize('frames,valid',[(16000,100),(16001,101),(480000,3000)])
def test_feature_mask_marks_actual_view_frames_without_truncation(frames,valid):
    from asr.processor import prepare_example
    from asr.contracts import UtteranceRow
    audio=np.zeros(frames,dtype=np.float32); audio[-1]=0.5
    before=audio.copy()
    example=prepare_example(UtteranceRow('s','current','s/current','audio/x.wav','hi'),audio,processor_fixture())
    assert example.input_features.shape==(1,80,3000)
    assert example.attention_mask.shape==(1,3000)
    assert example.attention_mask.sum().item()==valid
    assert example.attention_mask[0,:valid].eq(1).all()
    assert example.attention_mask[0,valid:].eq(0).all()
    np.testing.assert_array_equal(audio,before)


def test_label_boundary_nfc_decoder_start_and_eos_padding():
    from asr.processor import prepare_example,collate_examples
    from asr.contracts import UtteranceRow
    p=processor_fixture(); audio=np.zeros(16000,dtype=np.float32)
    row=UtteranceRow('s','current','s/current','audio/x.wav','n=443')
    full=prepare_example(row,audio,p)
    assert full.labels.shape==(1,447)
    assert full.labels[0,0].item()==2
    with pytest.raises(ValueError,match='447'):
        prepare_example(UtteranceRow('s','current','s/current','audio/x.wav','n=444'),audio,p)
    short=prepare_example(UtteranceRow('s','current','s/current','audio/x.wav','  Cafe\u0301  '),audio,p)
    assert p.tokenizer.last_text=='Café'
    batch=collate_examples([short,full])
    assert batch['labels'][0,short.labels.shape[1]-1].item()==0
    assert batch['labels'][0,short.labels.shape[1]:].eq(-100).all()


def test_feature_rejects_oversized_and_empty_audio():
    from asr.processor import prepare_example
    from asr.contracts import UtteranceRow
    p=processor_fixture(); row=UtteranceRow('s','current','s/current','audio/x.wav','hi')
    for frames in (0,480001):
        with pytest.raises(ValueError):
            prepare_example(row,np.zeros(frames,dtype=np.float32),p)
