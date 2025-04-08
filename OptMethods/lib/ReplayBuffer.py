
import numpy as np
import random
from operator import itemgetter
import torch

class Replay_buffer():
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        # batch = random.sample(self.storage, batch_size)
        # x, y, u, r, d = map(np.stack, zip(*batch))
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = map(np.stack, zip(*itemgetter(*ind)(self.storage)))

        return x,y,u,r,d
    
    def getEpisodeBatch(self, steps):

        if len(self.storage) == 0:
            return
        
        if len(self.storage) == self.max_size:
            idx = self.ptr % self.max_size
            if idx-steps-1 < 0:
                idxList = np.concatenate((np.arange(self.max_size-(steps+1-idx), self.max_size),np.arange(0,idx)))
            else:
                idxList = np.arange(idx-steps-1,idx)
        else:
            idxList = np.arange(len(self.storage)-steps-1,len(self.storage))
            
        batch = list(zip(*itemgetter(*idxList)(self.storage)))
        # if observation is numpy
        if isinstance(batch[0][0], np.ndarray):
            observationBatch = np.array(batch[0])
            observationNextBatch = np.array(batch[1])
            actionBatch = np.array(batch[2])
            rewardBatch = np.array(batch[3])
        else:
            observationBatch = torch.stack(batch[0]).cpu().data.numpy()
            observationNextBatch = torch.stack(batch[1]).cpu().data.numpy()
            actionBatch = torch.stack(batch[2]).cpu().data.numpy().flatten()
            rewardBatch = torch.stack(batch[3]).cpu().data.numpy().flatten()
        doneBatch = np.array(batch[4])

        return (observationBatch, observationNextBatch, actionBatch, rewardBatch, doneBatch)
