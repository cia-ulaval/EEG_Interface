
from tsai.all import *
import pandas as pd
import torch
torch.set_default_device('cuda') 

def cli():
    torch.set_default_device('cuda') 
    window_length = 4
    n_vars = 5
    seq_len = 100
    horizon = 1


    df = pd.read_csv('Data/EyesClosed/1.csv',index_col=0)
    print('input shape:', df.shape)
    X, y = SlidingWindow(window_length, stride=None, start=0, get_x=df.columns[:-1],  get_y=df.columns[-1], horizon=0, seq_first=True)(df)
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False)
    tfms  = [None, [Categorize()]]
    print(X)
    print(y)
    dsets = TSDatasets(torch.Tensor(X), torch.Tensor(y), tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0,device='cuda')
    dls.show_batch(sharey=True)
    model = InceptionTime(dls.vars, dls.c)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(25, lr_max=1e-3)
    learn.save('stage1')
    learn.recorder.plot_metrics()


if __name__ == '__main__':
    cli()