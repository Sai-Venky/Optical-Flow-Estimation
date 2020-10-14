import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.dataset import Dataset
import numpy as np

from models.optical_flow import OpticalFlow
from utils.config import opt

def run(**kwargs):

    '''
        Runs the optical flow model which includes the training and validation.
    '''
        
    opt._parse(kwargs)
    opt.number_gpus = torch.cuda.device_count()
    if opt.number_gpus == 0:
        opt.effective_batch_size = opt.batch_size
    else :
        opt.effective_batch_size = opt.batch_size * opt.number_gpus

    opt.cuda = torch.cuda.is_available()

    training_dataset = Dataset(opt.dataset_dir + 'training')
    training_dataloader = DataLoader(training_dataset, batch_size=opt.effective_batch_size, shuffle=True, num_workers=opt.number_workers)

    validation_dataset = Dataset(opt.dataset_dir + 'validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=opt.effective_batch_size, shuffle=True, num_workers=opt.number_workers)
    opticalFlow = OpticalFlow()
    if opt.cuda and opt.number_gpus == 1:
        opticalFlow = opticalFlow.cuda()

    if opt.load_path:
        opticalFlow.load(opt.load_path)
        
    best_err = 100000
    for epoch in range(opt.total_epochs):
        training_loss, batch_idex = opticalFlow.train(training_dataloader)
        
        validation_loss, batch_idex = opticalFlow.validate(validation_dataloader)

        is_best = False
        if validation_loss < best_err:
            best_err = validation_loss
            is_best = True

        opticalFlow.save(is_best)

if __name__ == '__main__':
    run()