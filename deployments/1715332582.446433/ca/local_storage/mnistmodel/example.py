from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


from .trainer import Trainer

@dataclass
class MnistModelConfig():
    inpsiz = 784 
    hidensiz = 500 
    numclases = 10
    numepchs =50
    bachsiz = 100
    l_r = 0.001 


class MnistModel(nn.Module):
    def __init__(self, inpsiz, hidensiz, numclases):
         super(MnistModel, self).__init__()
         self.inputsiz = inpsiz
         self.l1 = nn.Linear(inpsiz, hidensiz) 
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(hidensiz, numclases) 
    def forward(self, y):
         outp = self.l1(y)
         outp = self.relu(outp)
         outp = self.l2(outp)

         return outp

    def train_step(self, batch, criterion):
        x, y = batch
        logits = self(x)
        loss = criterion(logits, y)
        return {"model_outputs": logits}, {"loss": loss}

    def eval_step(self, batch, criterion):
        x, y = batch
        logits = self(x)
        loss = criterion(logits, y)
        return {"model_outputs": logits}, {"loss": loss}

    @staticmethod
    def get_criterion():
        return torch.nn.CrossEntropyLoss()

    
    

def main():
    config = MnistModelConfig()

    model = MnistModel(config.inpsiz, config.hidensiz,config.numclases)

    trainer = Trainer(
        config,
        model=model,
    )

    for epch in range(config.numepchs):
        trainer.one_step(epch)

        trainer.get_logs()

        trainer.get_last_checkpoint()

        #trainer.get_best_checkpoint()

    print(trainer.get_config())




if __name__ == "__main__":
    main()