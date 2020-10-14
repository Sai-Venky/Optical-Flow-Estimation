import torch
import torch.nn as nn
import math

class MultiScale(nn.Module):

    '''
        Computes the loss for the model.
        Uses L1 loss for multiscale computation
    '''

    def __init__(self, startScale = 4, numScales = 5, l_weight= 0.32):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.div_flow = 0.05

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]

    def forward(self, output, target):

        '''
            Computes the loss for the model.
            Uses L1 loss for multiscale computation
            Arguments:
                output      :- Precited flow
                target      :- Target flow
            Returns:
                lost of the loss value, epe value
        '''

        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*(torch.norm(output_ - target_,p=2,dim=1).mean())
                lossvalue += self.loss_weights[i]*(torch.abs(output_ - target_).mean())
            return [lossvalue, epevalue]
        else:
            epevalue += torch.norm(output-target,p=2,dim=1).mean()
            lossvalue = torch.abs(output - target).mean()
            return  [lossvalue, epevalue]

