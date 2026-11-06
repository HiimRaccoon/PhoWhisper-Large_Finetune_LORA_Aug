from types import SimpleNamespace
import torch


class ToyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A=torch.nn.Parameter(torch.tensor([[0.1]],dtype=torch.float32))
    def forward(self,input_features,labels,attention_mask=None):
        return SimpleNamespace(loss=((input_features@self.lora_A-labels.float())**2).mean())


def examples(k):
    from asr.contracts import PreparedExample
    return [PreparedExample(torch.tensor([[0.1+i*0.01]]),torch.ones(1,1,dtype=torch.long),
                            torch.tensor([[1.0]]),f's/{i}') for i in range(k)]


def bundle_for(model):
    from asr.model import TrainingBundle
    from asr.contracts import RuntimeProfile
    return TrainingBundle(model,['lora_A'],RuntimeProfile())
