import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeLayerMLP(nn.Module):
    def __init__(self, xDim, xMean, xStd, yDim, modelDimList, OptionDict={}):
        super(ThreeLayerMLP, self).__init__()

        self.l1 = nn.Linear(xDim, modelDimList[0])
        self.l2 = nn.Linear(modelDimList[0], modelDimList[1])
        self.l3 = nn.Linear(modelDimList[1], yDim)
        
        self.OptionDict = OptionDict
        self.LIMIT_Y = False
        for k, v in OptionDict.items():
            if k == 'yMax':
                self.yMax = v
                self.LIMIT_Y = True

        self.xmean = xMean
        self.xstd = xStd

        #self.model_name = modelName

    def forward(self, *x):
        x = torch.hstack(x)
        
        if not x.is_cuda:    
            x = x.cuda()

        x = (x-self.xmean)/self.xstd

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        if self.LIMIT_Y:
            x = self.yMax * torch.tanh(x)
        
        return x
