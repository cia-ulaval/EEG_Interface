import click
import lightning as L
import pandas as pd
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split

from eeg_interface.data_handler.demo import DemoDataset
from eeg_interface.models.demo import Dense, LitDense
from eeg_interface.utils.visualize import visualize_time_series


@click.command(help="Classification of eeg data demo")

@click.option(
    '--input-path', '-i', 
    type=click.Path(exists=True, file_okay=True, dir_okay=True), 
    default='./', 
    help='input data path'
)


def demo(input_path, ):
    demoDataset = DemoDataset(input_path)
    
    trainSet,testSet = random_split(demoDataset,[0.8,0.2])
    trainSet,valSet = random_split(trainSet,[0.8,0.2])
    
    train_loader = DataLoader(trainSet,batch_size=64,shuffle=True)
    val_loader = DataLoader(valSet,batch_size=64,shuffle=True)
    test_loader = DataLoader(testSet,batch_size=64,shuffle=True)
    
    dense = LitDense(Dense())
    
    mlf_logger = MLFlowLogger(experiment_name="DemoEEG", tracking_uri="http://localhost:5000",log_model=True,artifact_location='../MLFlowLog')
    
    trainer = L.Trainer(max_epochs=10,logger=mlf_logger)
    trainer.fit(model=dense,train_dataloaders=train_loader,val_dataloaders=val_loader)
    trainer.test(model=dense, dataloaders=test_loader)
    
    
    


if __name__ == '__main__':
    demo()