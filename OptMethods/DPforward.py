# Dynamic Programming backward algorithm

import numpy as np
from typing import Mapping
from multiprocessing import Pool
from itertools import repeat

class DPforward():

    def __init__(self, Env, dRes=1, vRes=0.2, aRes=0.1):
        
        # get number of steps and dt from Env
        N = Env.N
        dt = Env.dt
        
        # discretize states 
        dArr = np.arange(np.floor(np.min(Env.dp-Env.dmax)), np.ceil(np.max(Env.dp-Env.dmin)), dRes)
        vArr = np.arange(np.floor(Env.vmin), np.ceil(Env.vmax), vRes)
        aArr = np.arange(np.floor(Env.amin), np.ceil(Env.amax), aRes)
        tArr = Env.t
        
        # tolerance
        d0tol = max(0.1, np.average(np.diff(dArr))+1e-3)
        dftol = max(0.1, np.average(np.diff(dArr))+1e-3)
        v0tol = max(0.01, np.average(np.diff(vArr))+1e-3)
        vftol = max(0.1, np.average(np.diff(vArr))+1e-3)

        # try to vectorize things up
        # dList = [d0,d0,d0,...d1,d1,d1,...dn,dn,dn,...]
        # vList = [v0,v1,v2,...v0,v1,v2,...v0,v1,v2,...]
        # stateList = [[d0,v0], [d0,v1], ... [dn,vn]]  
        vList, dList = np.meshgrid(vArr, dArr)
        dList, vList = dList.flatten(), vList.flatten()
        stateList = np.zeros((dList.size,2))
        stateList[:,0] = dList
        stateList[:,1] = vList

        self.stateList = stateList
        self.dList = dList
        self.vList = vList

        # dpList = np.arange(np.floor(np.minimum(Env.dp)), np.ceil(np.maximum(Env.dp)), dRes)
        # self.dpList = dpList

        self.Env = Env
        self.tArr = tArr
        self.dArr = dArr
        self.vArr = vArr
        self.aArr = aArr
        self.dRes = dRes
        self.vRes = vRes
        self.aRes = aRes
        self.d0tol = d0tol
        self.dftol = dftol
        self.v0tol = v0tol
        self.vftol = vftol

        # reset
        self.reset()

    def runOpt(self):

        # reset
        self.reset()

        # short names
        # tArr, dArr, vArr, aArr = self.tArr, self.dArr, self.vArr, self.aArr
        # ValueMap, QvalueMap = self.ValueMap, self.QvalueMap
        Env = self.Env
        dp = Env.dp
        N = Env.N
        dOpt, vOpt, aOpt = self.dOpt, self.vOpt, self.aOpt
        info = self.info

        # final value
        # initlaize V(s)
        #
        # repeat until delta smaller than threshold
        #   delta = 0
        #   for each S
        #       v = V(s)
        #       V(s) = min_a (r+V(s'))
        #       delta = max(delta, v-V(s))
        # pi(s) = argmin_a (r+V(s))

        # V(s) initialization is done by reset() function

        # initialize training parameters
        tArr, dArr, vArr, aArr = self.tArr, self.dArr, self.vArr, self.aArr
        actionList = self.aArr
        stateList, dList, vList = self.stateList, self.dList, self.vList
        dpList = self.Env.dp

        #deltaList = np.array([np.inf]*self.dList.size)

        delta = np.inf
        deltaNoInfMax = np.inf
        thld = 0.2
        iIter = 0
        iMax = 500
        # while delta >= thld and i < iMax:
        # while (deltaNoInfMax >= thld or iIter < 1) and iIter < iMax:
        while iIter < 1:
            delta = 0
            deltaNoInfMax = 0

            # now with the iteration through dp, we essentially have the concept of time step again
            for kk, dp in enumerate(np.flip(dpList)):
                k = dpList.size-1-kk
                # print('optimizing iteration {}/{}, k {}'.format(iIter, iMax, k))

                print('optimizing step {}/{}, progress {:.4f}%'.format(k+1, N, (N-(k))/N*100))

                # check constraint to find out which states are valid, then only find values of these states, all other states will have inf value
                idxLegitState = np.where(self.constrCheck(stateList, k))[0]
                
                # for each legit state, update Value
                #   need to go through all possible action then find the smallest/best value
                # 
                # divide into several groups of state, action pairs then vectorize it
                ns = stateList[idxLegitState].size
                na = actionList.size
                chunkSize = np.floor(5*1e9/8/na) # allow 5Gb size of the array
                nchunk = np.ceil(ns/chunkSize)
                stateListChunk = np.split(stateList[idxLegitState], nchunk)
                for iChunk, sListChunk in enumerate(stateListChunk):
                    # get number of state in current chunk
                    nsInChunk = np.shape(sListChunk)[0]

                    # idx of each state back in the original stateList
                    idxInOrigStateList = idxLegitState[np.arange((iChunk)*chunkSize,(iChunk)*chunkSize+min(nsInChunk,chunkSize),1).astype(int)]

                    # store previous value
                    valuePrev = self.ValueMap[idxInOrigStateList,k]

                    # make vectorized list of state and corresponding action
                    # sList is aArr.size of S: [S, S, ..., S], [[d0,v0],[d0,v1],...[dn,vn], [d0,v0],[d0,v1],...[dn,vn]] 
                    # aList is S.size of aArr then transpose: [A, A, ..., A]^T, [a0,a0,...a0,a1,a1,...a1,...,an], each a_i is of size S.size
                    sList = np.tile(sListChunk, (aArr.size, 1))
                    aList = (np.tile(aArr, (nsInChunk, 1))).transpose().flatten()

                    # print('idxState {}'.format(idxState))

                    # deltaList[idxState] = 0

                    if k < N-1:
                        # get list of next state, which is not round to grid of state yet
                        stateNextRaw = Env.getNextState(sList, aList)

                        # conver to idx in stateList
                        # dList, vList, stateList, ValueMap[:,k] all same size
                        tmpIdxD = np.round((stateNextRaw[:,0]-dArr[0])/self.dRes)
                        tmpIdxV = np.round((stateNextRaw[:,1]-vArr[0])/self.vRes)
                        idx = (tmpIdxD*vArr.size + tmpIdxV).astype(int)
                        # if say state is alreay on the upper bound, those state will be discarded and not add to value list
                        idxValid = (tmpIdxD <= dArr.size-1) & (tmpIdxV <= vArr.size-1)
                        # idx = idx[idxValid]

                        # stateNext = stateList[idx]

                        # ValueMap corresponding to
                        #           dp0  dp1  ...
                        #   d0,v0   v0,0 v0,1 ...
                        #   d0,v1   v1,0 v1,1 ...
                        #   ...
                        r,_,_ = Env.getReward(sList, aList)
                        valueList = r + self.ValueMap[np.minimum(idx, dList.size-1),k+1]
                        valueList[~idxValid] = np.inf
                    else:
                        r,_,_ = Env.getReward(sList, aList)
                        valueList = r

                    valueListReshape = (np.reshape(valueList, (-1, nsInChunk))).transpose()
                    aListReshape = (np.reshape(aList, (-1, nsInChunk))).transpose()
                    idxMinValue = np.argmin(valueListReshape, axis=1)
                    # aOpt[k] = actionList[idxMinValue]

                    self.ValueMap[idxInOrigStateList,k] = valueListReshape[np.arange(0, nsInChunk),idxMinValue]
                    self.OptActionMap[idxInOrigStateList,k] = aListReshape[np.arange(0, nsInChunk),idxMinValue]
                    
                    # update condition, exclude nan (inf-inf)
                    valueDiff = valuePrev-self.ValueMap[idxInOrigStateList,k]

                    if np.all(np.isnan(valueDiff)):
                        delta = np.inf
                    else:
                        valueDiffMax = np.nanmax(np.abs(valueDiff))
                        delta = np.max([delta, valueDiffMax])

                    if valueDiff[np.isfinite(valueDiff)].size:
                        valueNoInfDiffMax = np.nanmax(np.abs(valueDiff[np.isfinite(valueDiff)]))
                        deltaNoInfMax = np.max([deltaNoInfMax, valueNoInfDiffMax])
                    else:
                        deltaNoInfMax = np.inf

            print('optimizing iteration {}/{}, delta {:.4f}, delta noInf {:.4f}, threshold {:.4f}'.format(iIter, iMax, delta, deltaNoInfMax, thld))
            print('\ttotal value {}, number of finite {}'.format(self.ValueMap.size, self.ValueMap[np.isfinite(self.ValueMap)].size))

            # update number of iteration
            iIter = iIter + 1


        # retrieve optimal values
        tOpt = self.tArr
        vv = np.empty(shape=(self.Env.N,))
        rr = np.empty(shape=(self.Env.N,))
        ii = np.empty(shape=(self.Env.N,))

        for k in range(0, Env.N):
            if k == 0:
                idx = np.argmin(self.ValueMap[:,k])       
            else:
                stateNextRaw = Env.getNextState(np.array([dOpt[k-1], vOpt[k-1]]), np.array([aOpt[k-1]]))
                tmpIdxD = np.round((stateNextRaw[0]-self.dArr[0])/self.dRes)
                tmpIdxV = np.round((stateNextRaw[1]-self.vArr[0])/self.vRes)
                idx = (tmpIdxD*self.vArr.size + tmpIdxV).astype(int)

            dOpt[k] = self.stateList[idx,0]
            vOpt[k] = self.stateList[idx,1]
            aOpt[k] = self.OptActionMap[idx,k]
            vv[k] = self.ValueMap[idx,k] 
            r,_,_ = Env.getReward(np.array([dOpt[k], vOpt[k]]), np.array([aOpt[k]]))
            rr[k] = r
            ii[k] = idx

        # store info
        info['dRes'] = self.dRes
        info['vRes'] = self.vRes
        info['aRes'] = self.aRes
        info['dOpt'] = dOpt
        info['vOpt'] = vOpt
        info['aOpt'] = aOpt
        info['valueOpt'] = vv
        info['rewardOpt'] = rr
        info['idxOpt'] = ii
        if delta < thld:
            info['status'] = 'optimal ValueMap found under threshold {}'.format(thld)
        elif iIter >= 500:
            info['status'] = 'maximum iteration reached {}'.format(iIter)

        return tOpt, dOpt, vOpt, aOpt, info
    

    # def __getValue(self, state):
    #     # now state = (d, v, dp)
    #     d = 



    # def __followingConstraintCheck(self, state):
    def constrCheck(self, stateList, k):
        if k == 0:
            d, v = stateList[:,0], stateList[:,1]
            return (d>=self.Env.d0-self.d0tol) & (d<=self.Env.d0+self.d0tol) & (v>=self.Env.v0-self.v0tol) & (v<=self.Env.v0+self.v0tol)
        elif k == self.Env.N-1:
            d, v = stateList[:,0], stateList[:,1]
            return (d>=self.Env.df-self.dftol) & (d<=self.Env.df+self.dftol) & (v>=self.Env.vf-self.vftol) & (v<=self.Env.vf+self.vftol)
        else:
            d, v = stateList[:,0], stateList[:,1]
            return (self.Env.dp[k] - d <= self.Env.dmax) & (self.Env.dp[k] - d >= self.Env.dmin)
    


    def reset(self):

        # ValueMap corresponding to
        #           dp0  dp1  ...
        #   d0,v0   v0,0 v0,1 ...
        #   d0,v1   v1,0 v1,1 ...
        #   ...
        dpList = self.Env.dp
        ValueMap =  np.ones(shape=(self.dList.size, dpList.size))*np.inf
        QvalueMap =  np.ones(shape=(self.dList.size, dpList.size))*np.inf
        OptActionMap =  np.ones(shape=(self.dList.size, dpList.size))*np.inf

        # array to store optimal acceleration
        dOpt = np.empty(shape=(self.Env.N,))
        vOpt = np.empty(shape=(self.Env.N,))
        aOpt = np.empty(shape=(self.Env.N,))

        info = {}

        self.ValueMap = ValueMap
        self.QvalueMap = QvalueMap
        self.OptActionMap = OptActionMap

        self.dOpt, self.vOpt, self.aOpt = dOpt, vOpt, aOpt
        self.info = info
