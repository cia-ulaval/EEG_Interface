import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
import mlflow
import torchmetrics

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(14 , 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.l1(x)
    
class LitDense(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.reshape(-1,1)
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.mse_loss(z, y)
        precision = torchmetrics.functional.precision(z,y,task='binary')
        recall = torchmetrics.functional.recall(z,y,task='binary')
        f1 = torchmetrics.functional.f1_score(z,y,task='binary')
        self.logger.log_metrics({'Trainloss':loss,
                                 'TrainPrecision':precision,
                                 'TrainRecall':recall,
                                 'TrainF1':f1},self.current_epoch)
        return loss
    
    def test_step(self, batch, batch_idx):

        x, y = batch
        y = y.reshape(-1,1)
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.mse_loss(z, y)
        precision = torchmetrics.functional.precision(z,y,task='binary')
        recall = torchmetrics.functional.recall(z,y,task='binary')
        f1 = torchmetrics.functional.f1_score(z,y,task='binary')
        self.logger.log_metrics({'TestLoss':loss,
                                 'TestPrecision':precision,
                                 'TestRecall':recall,
                                 'TestF1':f1},self.current_epoch)

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y = y.reshape(-1,1)
        x = x.view(x.size(0), -1)
        z = self.model(x)
        loss = F.mse_loss(z, y)
        precision = torchmetrics.functional.precision(z,y,task='binary')
        recall = torchmetrics.functional.recall(z,y,task='binary')
        f1 = torchmetrics.functional.f1_score(z,y,task='binary')
        self.logger.log_metrics({'ValidationLoss':loss,
                                 'ValidationPrecision':precision,
                                 'ValidationRecall':recall,
                                 'ValidationF1':f1},self.current_epoch)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer