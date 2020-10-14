import os
from os.path import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import time

from utils.config import opt
from data.utils import writeFlow, visulize_flow_file
from models.loss import MultiScale
from models.flow_net2SD import FlowNet2SD

class OpticalFlow(nn.Module):

    '''
        OpticalFlow is the base class which contains all the methods related to training, validation etc
    '''

    def __init__(self):
        super(OpticalFlow, self).__init__()
        self.model = FlowNet2SD()
        self.optimizer = self.get_optimizer()
        self.loss = MultiScale() 
        
    def forward(self, data, target, inference=False ):

        '''
            Return the two image and flow applicable for those images
            Arguments:
                data        :- the images
                target      :- the flow for the images
            Returns:
                loss_values :- The loss values computed between the computed and predicted flows
                output      :- the output value from flownet model
        '''
                
        output = self.model(data)

        loss_values = self.loss(output, target)

        if not inference :
            return loss_values
        else :
            return loss_values, output

    def get_optimizer(self):
        
        """
            returns Optimizer for Optical Flow
        """
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': opt.lr * 2}]
                else:
                    params += [{'params': [value], 'lr': opt.lr}]
        self.optimizer = torch.optim.Adam(params)
        return self.optimizer


    def train(self, data_loader, offset=0):

        '''
            Implements the train function for optical flow
            Arguments:
                data_loader :- Training Dataloader
                offset      :- offset parameter passed for tqdm
            Returns:
                the totalloss computed over the total batch size
        '''

        statistics = []
        total_loss = 0

        self.model.train()

        progress = tqdm(IteratorTimer(data_loader), ncols=120, total=np.minimum(len(data_loader), opt.effective_batch_size), smoothing=.9, miniters=1, leave=True, position=offset, desc = 'training')

        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):

            self.optimizer.zero_grad()

            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if opt.cuda and opt.number_gpus == 1:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]

            losses = self.forward(data[0], target[0], False)
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0]
            total_loss += loss_val.item()

            loss_val.backward()
                
            self.optimizer.step()

        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)


    def validate(self, data_loader, offset=0):


        '''
            Implements the validate function for optical flow
            There is also visualization of flow called in the same
            Arguments:
                data_loader :- Training Dataloader
                offset      :- offset parameter passed for tqdm
            Returns:
                the totalloss computed over the total batch size
        '''
        
        statistics = []
        total_loss = 0

        self.model.eval()

        progress = tqdm(IteratorTimer(data_loader), ncols=100, total=np.minimum(len(data_loader), opt.effective_batch_size), leave=True, position=offset, desc='Validation')

        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):


            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if opt.cuda and opt.number_gpus == 1:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]

            losses, output = self.forward(data[0], target[0], True)
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0]
            total_loss += loss_val.item()

            for i in range(opt.effective_batch_size):
                _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                writeFlow( join(opt.flow_folder, '%06d.flo'%(batch_idx * opt.effective_batch_size + i)),  _pflow)
                
                # You can comment out the plt block in visulize_flow_file() for real-time visualization
                visulize_flow_file(
                    join(opt.flow_folder, '%06d.flo' % (batch_idx * opt.effective_batch_size + i)),opt.flow_folder)
            

        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)        


    def save(self, is_best):

        '''
            Saves the model
            Arguments:
                is_best :- parameter to indicate if model is best
            Returns:
                the path of the model where its saved
        '''
        
        if is_best == False:
            return
        timestr = time.strftime('%m%d%H%M')
        save_path = 'checkpoints/opticalflow_%s' % timestr

        save_dict = dict()
        save_dict['model'] = self.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path):

        '''
            Loads the model
            Arguments:
                path :- load path
        '''
        
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        opt._parse(state_dict['config'])
        return self

class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n
