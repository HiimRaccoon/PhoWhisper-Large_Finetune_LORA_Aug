import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch


class GenerationBackend(torch.nn.Module):
    """Model boundary fixture: observes the real generation invocation contract."""
    def __init__(self,tokens=None):
        super().__init__()
        from transformers import GenerationConfig
        self.weight=torch.nn.Parameter(torch.zeros(1,dtype=torch.float16))
        self.generation_config=GenerationConfig(do_sample=True,num_beams=5,return_timestamps=True,
            forced_decoder_ids=[[1,999]],temperature=[0.0,0.2,0.4],eos_token_id=2)
        self.tokens=tokens if tokens is not None else [1,3,4,5,6,2]
        self.observed=None

    def generate(self,**kwargs):
        self.observed=(self.training,torch.is_grad_enabled(),kwargs)
        return SimpleNamespace(sequences=torch.tensor([self.tokens]))


class GenerationProcessor:
    def __init__(self):
        self.tokenizer=self
        self.feature_extractor=self
        self.seen_audio=None

    def get_vocab(self):
        return {'<|startoftranscript|>':1,'<|vi|>':3,'<|transcribe|>':4,
                '<|notimestamps|>':5,'<|translate|>':7,'<|endoftext|>':2}

    def __call__(self,audio,**kwargs):
        self.seen_audio=audio.copy()
        frames=np.arange(3000)*160<len(audio)
        return {'input_features':torch.zeros(1,80,3000),'attention_mask':torch.tensor(frames[None,:],dtype=torch.long)}

    def batch_decode(self,tokens,**kwargs): return ['model words' if 6 in tokens[0] else '']


def test_generation_uses_clean_audio_mask_and_locked_settings(tmp_path):
    from asr.inference import predict_utterance
    from test_outputs import output_fixture
    row=output_fixture(tmp_path)[0][0]
    model=GenerationBackend(); processor=GenerationProcessor()
    audio=np.linspace(-0.1,0.1,16000,dtype=np.float32)
    prediction=predict_utterance(model,processor,row,audio,output_fixture(tmp_path)[2])
    assert prediction['transcript']=='model words' and prediction['generation']['eos']
    training,grad,kwargs=model.observed
    assert not training and not grad and np.array_equal(processor.seen_audio,audio)
    assert kwargs['attention_mask'].sum()==100 and kwargs['input_features'].dtype==torch.float16
    generation=kwargs['generation_config']
    assert not generation.do_sample and generation.num_beams==1
    assert generation.language=='vi' and generation.task=='transcribe'
    assert generation.max_new_tokens==440 and not generation.return_timestamps
    assert generation.forced_decoder_ids is None and kwargs['temperature']==0.0
    assert kwargs['return_dict_in_generate'] is True
    assert prediction['generation']['generated_tokens']==2  # content and EOS, after four prefix tokens


@pytest.mark.parametrize('tokens,reason', [([1,3,4,5,2],'empty'),([1,3,4,5]+[6]*440,'limit')])
def test_generation_rejects_empty_or_limit_without_eos(tmp_path,tokens,reason):
    from asr.inference import predict_utterance,GenerationFailure
    from test_outputs import output_fixture
    rows,_,exported=output_fixture(tmp_path)
    with pytest.raises(GenerationFailure) as error:
        predict_utterance(GenerationBackend(tokens),GenerationProcessor(),rows[0],np.zeros(16000,dtype=np.float32),exported)
    assert error.value.evidence['reason']==reason
    assert 'SECRET' not in str(error.value.evidence)


def test_real_tiny_whisper_preserves_eos_evidence(local_config):
    from asr.inference import predict_utterance,GenerationFailure
    from test_export import tiny_export_components
    from test_outputs import output_fixture
    bundle,processor=tiny_export_components(local_config)
    rows,_,exported=output_fixture(local_config.paths.project_root)
    model=bundle.model.merge_and_unload().half().eval()
    model.generation_config.sequence_bias={(model.config.eos_token_id,):100000.0}
    model.generation_config.begin_suppress_tokens=[]
    with pytest.raises(GenerationFailure) as error:
        predict_utterance(model,processor,rows[0],np.zeros(16000,dtype=np.float32),exported)
    assert error.value.evidence['reason']=='empty' and error.value.evidence['eos'] is True
    assert error.value.evidence['generated_tokens']==1


def test_inference_commands_reject_missing_artifact_before_work(local_config,capsys):
    from asr.cli import main
    config=str(local_config.paths.project_root/'configs/official.yaml')
    missing=str(local_config.paths.project_root/'missing')
    assert main(['infer-train','--config',config,'--model',missing,'--manifest',missing,'--output',missing])==1
    assert main(['validate-output','--config',config,'--model',missing,'--manifest',missing,
                 '--utterances',missing,'--sources',missing])==1
