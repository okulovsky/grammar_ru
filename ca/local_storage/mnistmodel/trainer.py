import torch
import torch.nn as nn
from pathlib import Path 
import copy
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Trainer():
    def __init__(self, config,  model):
        self.config = config
        self.model = model
        self.logs = []
        self.best_loss = 1
        self.train_data=self.get_data_loader(self.config, "data/", False, None, None, None)
        self.criter = self.model.get_criterion()
        self.optim = torch.optim.Adam(self. model.parameters(), lr=self.config.l_r)
        self.nttlstps = len(self.train_data)
        

    def get_data_loader(
        self, config, assets, is_eval, samples, verbose, num_gpus, rank=0
    ):  # pylint: disable=unused-argument
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(os.getcwd(), train=not is_eval, download=True, transform=transform)
        dataset.data = dataset.data[:256]
        dataset.targets = dataset.targets[:256]
        dataloader = DataLoader(dataset, batch_size=config.bachsiz)
        return dataloader
    

    def one_step(self,epoch):
        for x, (imgs, lbls) in enumerate(self.train_data): 
                imgs = imgs.reshape(-1, 28*28)
                labls = lbls

                outp = self.model(imgs)
                losses = self.criter(outp, lbls)

                self.optim.zero_grad()
                losses.backward()
                self.optim.step() 
            
                self.logs.append(f'Epochs [{epoch+1}/{self.config.numepchs}], Step[{x+1}/{self.nttlstps}], Losses: {losses.item():.4f} \n')

                if losses.item() < self.best_loss:
                    self.best_val_loss = losses.item()
                    self.best_model = copy.deepcopy(self.model)

        if epoch % 10 == 0:
                self.last_model = copy.deepcopy(self.model)

            
        
    def fit(self):   
        for epoch in range(self.config.numepchs):
            for x, (imgs, lbls) in enumerate(self.train_data): 
                imgs = imgs.reshape(-1, 28*28)
                labls = lbls

                outp = self.model(imgs)
                losses = self.criter(outp, lbls)

                self.optim.zero_grad()
                losses.backward()
                self.optim.step() 
            
                self.logs.append(f'Epochs [{epoch+1}/{self.config.numepchs}], Step[{x+1}/{self.nttlstps}], Losses: {losses.item():.4f} \n')

                if losses.item() < self.best_loss:
                    self.best_val_loss = losses.item()
                    self.best_model = copy.deepcopy(self.model)

            


    def get_logs(self):
        f  = open(Path('grammar_ru/tg/ca/local storage/data/logs.txt'),'w')
        f.writelines(self.logs)

        return f
    
    def get_last_checkpoint(self):
        torch.save(self.last_model.state_dict(), 'grammar_ru/tg/ca/local storage/data/last_model.pt')


    def get_best_checkpoint(self):
        torch.save(self.best_model.state_dict(), 'grammar_ru/tg/ca/local storage/data/best_model.pt')


    def get_config(self):
        return self.config



 
    

        


        

        