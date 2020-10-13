import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from utils.config import opt
from models.loss import MultiScale

class OpticalFlow(nn.Module):
    def __init__(self, args):
        super(OpticalFlow, self).__init__()
        self.model = model_map[opt.model]
        self.optimizer = self.get_optimizer()
        self.loss = MultiScale() 
        
    def forward(self, data, target, inference=False ):
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


    def train(data_loader, model, optimizer, is_validate=False, offset=0):
        statistics = []
        total_loss = 0

        model.train()

        progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=np.minimum(len(data_loader), opt.effective_batch_size), smoothing=.9, miniters=1, leave=True, position=offset, desc = 'training')

        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):

            optimizer.zero_grad()

            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if opt.cuda and opt.number_gpus == 1:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]

            losses = model(data[0], target[0])
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0]
            total_loss += loss_val.item()

            loss_val.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), opt.gradient_clip)

            params = list(model.parameters())
            for i in range(len(params)):
                param_copy[i].grad = params[i].grad.clone().type_as(params[i]).detach()
                param_copy[i].grad.mul_(1./opt.loss_scale)
            optimizer.step()
            for i in range(len(params)):
                params[i].data.copy_(param_copy[i].data)

        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)


    def validate(data_loader, model, optimizer, is_validate=False, offset=0):
        statistics = []
        total_loss = 0

        model.eval()

        progress = tqdm(tools.IteratorTimer(data_loader), ncols=100, total=np.minimum(len(data_loader), args.effective_batch_size), leave=True, position=offset, desc='Validation')

        last_log_time = progress._time()
        for batch_idx, (data, target) in enumerate(progress):


            data, target = [Variable(d) for d in data], [Variable(t) for t in target]
            if opt.cuda and opt.number_gpus == 1:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]

            losses = model(data[0], target[0])
            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0]
            total_loss += loss_val.item()


        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)        


    def save(self, is_best):
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
