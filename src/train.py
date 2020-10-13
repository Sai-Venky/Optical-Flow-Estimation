import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.dataset import Dataset
import numpy as np

from models.optical_flow import OpticalFlow


def run(**kwopts):
    
    opt.effective_batch_size = opts.batch_size * opts.number_gpus

    training_dataset = Dataset(opt.dataset_dir + 'training')
    training_dataloader = DataLoader(training_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=opt.num_workers)

    validation_dataset = Dataset(opt.dataset_dir + 'validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=opt.num_workers)
    opticalFlow = OpticalFlow()


    best_err = 100000
    for epoch in progress:
        training_loss = opticalFlow.train(data_loader=training_dataloader, model=opticalFlow, optimizer=optimizer)
        
        validation_loss = opticalFlow.validate(data_loader=validation_dataloader, model=opticalFlow, optimizer=optimizer)

        is_best = False
        if validation_loss < best_err:
            best_err = validation_loss
            is_best = True

        opticalFlow.save(is_best)

if __name__ == '__main__':
    run()