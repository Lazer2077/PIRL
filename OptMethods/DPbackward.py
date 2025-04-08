
# Dynamic Programming backward algorithm

import numpy as np
from typing import Mapping
from multiprocessing import Pool
from itertools import repeat
import torch

class DPbackward():

    def __init__(self, Env, dRes=1, vRes=0.2, aRes=0.1, SELECT_MIN_MAX='min'):
        
        # get number of steps and dt from Env
        N = Env.N+1
        dt = Env.dt
        
        # discretize states 
        dArr = np.arange(np.floor(np.min(Env.dp-Env.dmax)), np.ceil(np.max(Env.dp-Env.dmin)), dRes)
        vArr = np.arange(np.floor(Env.vmin), np.ceil(Env.vmax), vRes)
        aArr = np.arange(np.floor(Env.amin), np.ceil(Env.amax), aRes)
        tArr = Env.t

        self.N = N
        self.Env = Env
        self.tArr = tArr
        self.dArr = dArr
        self.vArr = vArr
        self.aArr = aArr
        self.dRes = dRes
        self.vRes = vRes
        self.aRes = aRes

        self.SELECT_MIN_MAX = SELECT_MIN_MAX

        # reset
        self.reset()

    def runOpt(self):
        # initialize terminal cost 
        # k = T-1
        
        # for each k
        #    for each state S_t
        #    max_a_t = -inf
        #    for each action a_t
        #        calculate v = c(S_t, a_t)+V_t+1(S_t+1)
        #        if v > max_a_t
        #            max_a_t = v
        #    V_t(S_t) = max_a_t(c(S_t, a_t)+V_t+1(S_t+1))
        #    store all of the V_t(S_t)
        # decrement k

        # reset
        self.reset()

        # short names
        # tArr, dArr, vArr, aArr = self.tArr, self.dArr, self.vArr, self.aArr
        # ValueMap, QvalueMap = self.ValueMap, self.QvalueMap
        Env = self.Env
        dp = Env.dp

        # tolerance
        d0tol = max(0.1, np.average(np.diff(self.dArr))+1e-3)
        dftol = max(0.1, np.average(np.diff(self.dArr))+1e-3)
        v0tol = max(0.01, np.average(np.diff(self.vArr))+1e-3)
        vftol = max(0.1, np.average(np.diff(self.vArr))+1e-3)
    
        
        # try to vectorize things up
        vList, dList = np.meshgrid(self.vArr, self.dArr)
        dList, vList = dList.flatten(), vList.flatten()
        stateList = np.zeros((dList.size,2))
        stateList[:,0] = dList
        stateList[:,1] = vList

        self.stateList = stateList
        self.dList = dList
        self.vList = vList

        # final value
        def finalValueCheck(stateList):
            d, v = stateList[:,0], stateList[:,1]
            return (d>=Env.df-dftol) & (d<=Env.df+dftol) & (v>=Env.vf-vftol) & (v<=Env.vf+vftol)
        self.__basicIterLoop(self.N-1, finalValueCheck)
        
        # do the loop till k = 1
        def constrCheck(stateList, k):
            d, v = stateList[:,0], stateList[:,1]
            return (dp[k] - d <= Env.dmax) & (dp[k] - d >= Env.dmin)
        for k in range(self.N-2, 0, -1):
            self.__basicIterLoop(k, constrCheck, k)

        # initial value
        def initValueCheck(stateList):
            d, v = stateList[:,0], stateList[:,1]
            return (d>=Env.d0-d0tol) & (d<=Env.d0+d0tol) & (v>=Env.v0-v0tol) & (v<=Env.v0+v0tol)
        self.__basicIterLoop(0, initValueCheck)

        # retrieve optimal value
        
    def retrieveOptValue(self):
        dOpt, vOpt, aOpt = self.dOpt, self.vOpt, self.aOpt
        Env = self.Env
        info = self.info
        
        tOpt = self.tArr
        vv = np.empty(shape=(self.N,))
        rr = np.empty(shape=(self.N,))
        ii = np.empty(shape=(self.N,))
        vvOpt = np.empty(shape=(self.N,self.N))
        for k in range(0, self.N):
            if k == 0:
                # if self.SELECT_MIN_MAX == 'max':
                #     idx = np.argmax(self.ValueMap[tOpt[k]])         
                # else:
                #     idx = np.argmin(self.ValueMap[tOpt[k]])    
                idx = np.linalg.norm(np.abs(self.stateList-np.array((Env.d0,Env.v0))),axis=1).argmin() 
                #stateNextRaw = self.stateList[idx,:]
            else:
                stateNextRaw = Env.getNextState(torch.FloatTensor([dOpt[k-1], vOpt[k-1]]), torch.FloatTensor([aOpt[k-1]]))
                stateNextRaw = stateNextRaw.data.numpy()[0]
                tmpIdxD = np.round((stateNextRaw[0]-self.dArr[0])/self.dRes)
                tmpIdxV = np.round((stateNextRaw[1]-self.vArr[0])/self.vRes)
                idx = (tmpIdxD*self.vArr.size + tmpIdxV).astype(int)

            dOpt[k] = self.stateList[idx,0]
            vOpt[k] = self.stateList[idx,1]
            aOpt[k] = self.OptActionMap[tOpt[k]][idx]
            vv[k] = self.ValueMap[tOpt[k]][idx] 
            r,_,_ = Env.getReward(torch.FloatTensor([dOpt[k], vOpt[k]]), torch.FloatTensor([aOpt[k]]), k=k)
            rr[k] = r.item()
            ii[k] = idx
            vvOpt[k,:] = np.array([v[idx] for v in self.ValueMap.values()])# this is map given each optimal state, then give its value over time. a N-by-N matrix

        # store info
        info['dRes'] = self.dRes
        info['vRes'] = self.vRes
        info['aRes'] = self.aRes
        # info['dList'] = self.stateList[:,0]
        # info['vList'] = self.stateList[:,1]
        info['dOpt'] = dOpt
        info['vOpt'] = vOpt
        info['aOpt'] = aOpt
        info['valueOpt'] = vv
        info['rewardOpt'] = rr
        info['idxOpt'] = ii
        info['ValueMap'] = vvOpt

        return tOpt, dOpt, vOpt, aOpt, vv, vv, rr, info
    

    def __basicIterLoop(self, k, constrCheck, *args):
        # short names
        tArr, dArr, vArr, aArr = self.tArr, self.dArr, self.vArr, self.aArr
        ValueMap, QvalueMap, OptActionMap = self.ValueMap, self.QvalueMap, self.OptActionMap
        Env = self.Env
        N = self.N
        # _, _, aOpt = self.dOpt, self.vOpt, self.aOpt
        stateList, dList, vList = self.stateList, self.dList, self.vList
        actionList = aArr

        print('optimizing step {}/{}, progress {:.4f}%'.format(k+1, N, (N-(k))/N*100))

        Env.k = k

        # initialization
        if 0: #self.dRes <= 0.05 or self.vRes <= 0.05  or self.aRes <= 0.05:
            if self.SELECT_MIN_MAX == 'max':
                self.ValueMap[self.tArr[k]] = np.array([-np.inf]*self.dList.size)
            else:
                self.ValueMap[self.tArr[k]] = np.array([np.inf]*self.dList.size)
            self.OptActionMap[self.tArr[k]] = np.array([np.nan]*self.dList.size)
            # for state in stateList:
            #     ValueMap[tArr[k]][state.tobytes()] = np.inf 

            idxLegitState = np.where(constrCheck(self.stateList, *args))[0]

            for i, state in enumerate(self.stateList[idxLegitState]):
                self.stateForLoop(self, k, state, idxLegitState[i])

                # if k < N-1:
                #     stateNextRaw = Env.getNextState(state, actionList)

                #     # stateNext = np.array([np.array([dArr[(np.abs(dArr - s[0])).argmin()], \
                #     #                                 vArr[(np.abs(vArr - s[1])).argmin()]]) for s in stateNextRaw])
                #     tmpIdxD = np.round((stateNextRaw[:,0]-dArr[0])/self.dRes)
                #     tmpIdxV = np.round((stateNextRaw[:,1]-vArr[0])/self.vRes)
                #     idx = (tmpIdxD*vArr.size + tmpIdxV).astype(int)
                #     # if say state is alreay on the upper bound, those state will be discarded and not add to value list
                #     idxValid = (tmpIdxD <= dArr.size-1) & (tmpIdxV <= vArr.size-1)
                #     idx = idx[idxValid]
                #     # print('i {}'.format(i))
                #     stateNext = stateList[idx]

                #     if idx.size == 0:
                #         continue
                #     valueList = Env.getReward(state, actionList[idxValid]) + ValueMap[tArr[k+1]][idx]
                # else:
                #     valueList = Env.getReward(state, actionList)

                # idxMinValue = np.argmin(valueList)
                # # aOpt[k] = actionList[idxMinValue]
                # ValueMap[tArr[k]][idxLegitState[i]] = valueList[idxMinValue]
                # OptActionMap[tArr[k]][idxLegitState[i]] = actionList[idxMinValue]

        else:
            ##########################
            ##########################
            # try to vectorize further the state and action loop

            # initialization
            if self.SELECT_MIN_MAX == 'max':
                ValueMap[tArr[k]] = np.array([-np.inf]*dList.size)
            else:
                ValueMap[tArr[k]] = np.array([np.inf]*dList.size)
            OptActionMap[tArr[k]] = np.array([np.nan]*dList.size)
            # for state in stateList:
            #     ValueMap[tArr[k]][state.tobytes()] = np.inf 

            #idxLegitState = np.where(constrCheck(stateList, *args))[0]
            idxLegitState = np.arange(0,np.shape(stateList)[0])

            # aList, sList = np.meshgrid(actionList, stateList[idxLegitState])
            # sList, aList = sList.flatten(), aList.flatten()
            sList = np.tile(stateList[idxLegitState], (aArr.size, 1))
            aList = (np.tile(aArr, (idxLegitState.size, 1))).transpose().flatten()

            if k < N-1:
                stateNextRaw = Env.getNextState(torch.FloatTensor(sList), torch.FloatTensor(aList))

                stateNextRaw = stateNextRaw.data.numpy()
                # conver to idx in stateList
                # dList, vList, stateList, ValueMap[tArr[k+1]] all same size
                tmpIdxD = np.round((stateNextRaw[:,0]-dArr[0])/self.dRes)
                tmpIdxV = np.round((stateNextRaw[:,1]-vArr[0])/self.vRes)
                idx = (tmpIdxD*vArr.size + tmpIdxV).astype(int)
                # if say state is alreay on the upper bound, those state will be discarded and not add to value list
                idxValid = (tmpIdxD <= dArr.size-1) & (tmpIdxV <= vArr.size-1)
                # idx = idx[idxValid]

                # if idx.size == 0:
                #     continue
                r,_,_ = Env.getReward(torch.FloatTensor(sList), torch.FloatTensor(aList))
                r = r.data.numpy()
                valueList = r + ValueMap[tArr[k+1]][np.minimum(idx, dList.size-1)]
                if self.SELECT_MIN_MAX == 'max':
                    valueList[~idxValid] = -np.inf
                else:
                    valueList[~idxValid] = np.inf
                # ttdd = np.round((sList[:,0]-dArr[0])/self.dRes)
                # ttvv = np.round((sList[:,1]-vArr[0])/self.vRes)
                # iii = (ttdd*vArr.size + ttvv).astype(int)
            else:
                valueList,_,_ = Env.getReward(torch.FloatTensor(sList), torch.FloatTensor(aList))
                valueList = valueList.data.numpy()

            valueListReshape = (np.reshape(valueList, (-1, idxLegitState.size))).transpose()
            aListReshape = (np.reshape(aList, (-1, idxLegitState.size))).transpose()
            if self.SELECT_MIN_MAX == 'max':
                idxOptValue = np.argmax(valueListReshape, axis=1)
            else:
                idxOptValue = np.argmin(valueListReshape, axis=1)
            # aOpt[k] = actionList[idxMinValue]
            ValueMap[tArr[k]][idxLegitState] = valueListReshape[np.arange(0, idxLegitState.size),idxOptValue]
            OptActionMap[tArr[k]][idxLegitState] = aListReshape[np.arange(0, idxLegitState.size),idxOptValue]

            #print('sum value {}, sum opt {}'.format(sum(ValueMap[tArr[k]][idxLegitState]), sum(OptActionMap[tArr[k]][idxLegitState])))

        if not np.any(np.isfinite(self.ValueMap[self.tArr[k]])):
            pass

        # now write back
        self.ValueMap = ValueMap
        self.QvalueMap = QvalueMap
        self.OptActionMap = OptActionMap

    @staticmethod
    def stateForLoop(self, k, state, idxState):

        # short names
        tArr, dArr, vArr, aArr = self.tArr, self.dArr, self.vArr, self.aArr
        ValueMap, QvalueMap, OptActionMap = self.ValueMap, self.QvalueMap, self.OptActionMap
        Env = self.Env
        N = self.N
        _, _, aOpt = self.dOpt, self.vOpt, self.aOpt
        stateList, dList, vList = self.stateList, self.dList, self.vList
        actionList = aArr

        # for i, state in enumerate(stateList[idxLegitState]):
        if k < N-1:
            stateNextRaw = Env.getNextState(torch.FloatTensor(state), torch.FloatTensor(actionList))
            stateNextRaw = stateNextRaw.data.numpy()
            # stateNext = np.array([np.array([dArr[(np.abs(dArr - s[0])).argmin()], \
            #                                 vArr[(np.abs(vArr - s[1])).argmin()]]) for s in stateNextRaw])
            tmpIdxD = np.round((stateNextRaw[:,0]-dArr[0])/self.dRes)
            tmpIdxV = np.round((stateNextRaw[:,1]-vArr[0])/self.vRes)
            idx = (tmpIdxD*vArr.size + tmpIdxV).astype(int)
            # if say state is alreay on the upper bound, those state will be discarded and not add to value list
            idxValid = (tmpIdxD <= dArr.size-1) & (tmpIdxV <= vArr.size-1)
            idx = idx[idxValid]
            # print('i {}'.format(i))
            stateNext = stateList[idx]

            if idx.size == 0:
                # print('returned')
                return
            r,_,_ = Env.getReward(torch.FloatTensor(state), torch.FloatTensor(actionList[idxValid]))
            r = r.data.numpy()
            valueList = r + ValueMap[tArr[k+1]][idx]
        else:
            valueList,_,_ = Env.getReward(torch.FloatTensor(state), torch.FloatTensor(actionList))
            valueList = valueList.data.numpy()

        if self.SELECT_MIN_MAX == 'max':
            idxOptValue = np.argmax(valueList)
        else:
            idxOptValue = np.argmin(valueList)
        # aOpt[k] = actionList[idxMinValue]
        ValueMap[tArr[k]][idxState] = valueList[idxOptValue]
        OptActionMap[tArr[k]][idxState] = actionList[idxOptValue]

        # now write back
        self.ValueMap = ValueMap
        self.QvalueMap = QvalueMap
        self.OptActionMap = OptActionMap


    def reset(self):
        ValueMap = {}
        QvalueMap = {}
        OptActionMap = {}
        for t in self.tArr:
            ValueMap[t] = np.empty(0)
            QvalueMap[t] = np.empty(0)
            OptActionMap[t] = np.empty(0)

        # array to store optimal acceleration
        dOpt = np.empty(shape=(self.N,))
        vOpt = np.empty(shape=(self.N,))
        aOpt = np.empty(shape=(self.N,))

        info = {}

        self.ValueMap = ValueMap
        self.QvalueMap = QvalueMap
        self.OptActionMap = OptActionMap

        self.dOpt, self.vOpt, self.aOpt = dOpt, vOpt, aOpt
        self.info = info