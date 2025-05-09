import numpy as np
import pandas as pd
import scipy.io
import os
import random
import torch
import h5py

ENABLE_DEBUG = True

def TerminalReward(state,dp_final,vp_final):
    # get terminal reward
    # state: [s,v,a,dp,k]
    # action: [a]
    # dp: [dp]
    # vp: [vp]
    # get terminal reward
    w5 = 10**0
    w6 = 10**1
    ht = 1.5
    dmin = 1
    if len(state.shape) == 1:
        df = dp_final-state[0]
        vf = vp_final-state[1]
    else:
        df = dp_final-state[:,0]
        vf = vp_final-state[:,1]
    dsafe = vf*ht + dmin 
    reward = w5*(df-dsafe)**2 +w6*(vf)**2
    return reward
    

class SimpleSpeed():
    def __init__(self, dataPath, SELECT_PREC_ID=None, SELECT_OBSERVATION='state', options={}):
        self.Debug = {} # debug variable dict
        # vehicle parameters
        m = 2000
        g = 9.81
        mu = 0.01
        theta_0 = 0
        C_wind = 0.3606
        
        C_bat = 216000
        phi = 0 
        Vehicle_Rho= 1.205
        Vehicle_Af=  2.2
        Vehicle_Cd=  0.306
        R_bat = 0.07411328
        kw = Vehicle_Rho* Vehicle_Af*Vehicle_Cd
        n = 24
        U_oc = 380.4960
        p1 = mu*m*g*np.cos(theta_0) + m*g*np.sin(theta_0)
        p2 = C_wind
        p3 = m
        self.Veh = {'m': m,
                    'g': g,
                    'mu': mu,
                    'theta_0': theta_0,
                    'C_wind': C_wind,
                    'C_bat': C_bat,
                    'phi': phi,
                    'kw': kw,
                    'R_bat': R_bat,
                    'U_oc': U_oc,
                    'n': n,
                    'p1': p1,
                    'p2': p2,
                    'p3': p3,
                  }

        # create a dummy preceding vehicle speed
        self.dt = 0.1
            
        # objective function weights
        self.w1 = 10**1.4 # acc  weight  
        self.w2 = 10**-3; # power weight 
        self.w3 = 10**0; # soft car following s1 df_max
        self.w4 = 10**0; # soft car following s2 df_min 
        self.w5 = 10**0; # terminal d
        self.w6 = 10**1; # terminal v
             
        # constraints
        self.dmax = 80
        self.dmin = 1
        self.ht = 1.5
        self.vmax = 25
        self.vmin = 0
        self.umax = 3
        self.umin = -3
        
        self.dataPath = dataPath
        self.SELECT_OBSERVATION = SELECT_OBSERVATION
        if 'EnableOldFashion' in options.keys():
            self.OLD_FASHION = options['EnableOldFashion']
        else:
            self.OLD_FASHION = False
        self.reset(options=options)



    def __GetLagrangeCoeff(self, n, x, y):
        L = [1]*(n+1)
        Lbasis = [1]*(n+1)

        # try to get combined coefficients
        for k in range(0, n+1): # start the outer loop through the data values for x
            
            for kk in range(0, (k)): # start the inner loop through the data values for x (if k = 0 this loop is not executed)
                L[k] = np.polymul(L[k],[1/(x[k]-x[kk]), - x[kk]/(x[k]-x[kk])]) # see the Lagrange interpolating polynomials
            
            for kk in range(k+1, n+1): # start the inner loop through the data values (if k = n this loop is not executed)
                L[k] = np.polymul(L[k],[1/(x[k]-x[kk]), - x[kk]/(x[k]-x[kk])]) # see the Lagrange interpolating polynomials

            pass
            Lbasis[k] = L[k]
            L[k] = y[k]*L[k]

        L = np.sum(np.array(L), axis=0)
        return L, Lbasis

    #def updatePrecedingVehicle(self, SELECT_PREC_ID=None, DATA_FILTER=None, IS_INIT=False, T_BEG=None, T_HORIZON=None, INIT_STATE=None):
    def updatePrecedingVehicle(self, options={}):
        # parser options
        if 'selectPrecedingId' in options.keys():
            SELECT_PREC_ID = options['selectPrecedingId']
        else:
            SELECT_PREC_ID = None
        if 'dataFilter' in options.keys():
            DataFilterFunc = options['dataFilter']
        else:
            DataFilterFunc = None
        if 'tBeg' in options.keys():
            T_BEG = options['tBeg']
        else:
            T_BEG = None
        if 'tHorizon' in options.keys():
            T_HORIZON = options['tHorizon']
        else:
            T_HORIZON = None
        if 'InitialState' in options.keys():
            INIT_STATE = options['InitialState']
        else:
            INIT_STATE = None       

        if 'manualPrecedingVehicleData' in options.keys(): 
            PrecInfo = options['manualPrecedingVehicleData']
        else:
            PrecInfo = None

        ControlMode = {'StopLine': False, 
                       'Terminate': False}

        if SELECT_PREC_ID is None:
            RAND_VEH = True
        else:
            RAND_VEH = False

        if self.OLD_FASHION:
            self.TrafficData = scipy.io.loadmat(self.dataPath)
            # remove not needed keys
            del self.TrafficData['__header__']
            del self.TrafficData['__version__']
            del self.TrafficData['__globals__']
            self.vehNames = sorted(self.TrafficData)
        else:
            # print(self.dataPath)
            self.TrafficData = h5py.File(self.dataPath, 'r')
            self.vehNames = np.array(sorted(self.TrafficData))[1:]

            # randomdize vehicle
            # if speed is almost all zero, we want to skip it to next time or speed
            vpSum = 0
            vpAvrg = 0
            it = 0
            NOT_VALID = True
            while NOT_VALID: #vpAvrg < 2:

                if PrecInfo is not None:
                    time = PrecInfo['t']
                    distance = PrecInfo['d']
                    speed = PrecInfo['v']
                    vehId = PrecInfo['id']
                elif self.OLD_FASHION:
                    if RAND_VEH:
                        vehId = random.sample(self.vehNames, 1)[0] #round(np.random.uniform(3,self.nVehicle))
                        SELECT_PREC_ID = int(vehId[4:])
                    else:
                        vehId = 'veh_{}'.format(SELECT_PREC_ID)

                    time = self.TrafficData[vehId][0][0][0][0]
                    distance = self.TrafficData[vehId][0][0][3][0]
                    speed = self.TrafficData[vehId][0][0][4][0]
                else:
                    # we do a find every time Env.reset(), this is still ok for tens of thousands of data
                    if RAND_VEH:
                        if DataFilterFunc is None:
                            vehId = np.random.choice(self.vehNames)
                        else:
                            self.VehNamesFiltered = DataFilterFunc(self.vehNames) # get the vehicle sthat satisfy filter condition
                            vehId = np.random.choice(self.VehNamesFiltered)
                        SELECT_PREC_ID = int(vehId.split('_')[1])
                    else:
                        vehId = SELECT_PREC_ID

                    time = np.array(self.TrafficData[vehId]['time']).reshape(-1)
                    distance = np.array(self.TrafficData[vehId]['distance']).reshape(-1)
                    speed = np.array(self.TrafficData[vehId]['speed']).reshape(-1)

                # randomdize time
                if T_HORIZON is None:
                    tHorizon = 15
                else:
                    tHorizon = T_HORIZON

                # normalize time first
                if T_BEG is None:
                    tBegSel = round(np.random.uniform(time[0], time[-1]-tHorizon))
                else:
                    #tBegSel = time[0]+2 # time[0]+14
                    tBegSel = time[0]+T_BEG
                tBeg = np.minimum(np.maximum(tBegSel, time[0]), time[-1]-tHorizon)
                tEnd = tBeg+tHorizon 
                idx = np.where((time >= tBeg - 1e-7) & (time <= tEnd + 1e-7))[0]

                # its possible the cycle is too short,
                # OR vehicle change lane and come back, if so, skip
                if len(idx) != int(tHorizon/self.dt+1):
                    continue

                tBegIdx = idx[0]

                t = time[idx]-time[idx[0]]+0.1 # add 0.1 so that initial polynomial states are non-zero
                dp = distance[idx]-distance[idx[0]]+0.1 
                vp = speed[idx]
                ap = np.hstack((np.diff(vp),np.array([0])))/self.dt
                
                # if speed is almost all zero, we want to skip it to next time or speed
                vpSum = np.sum(vp)
                vpAvrg = np.average(vp)

                it = it+1

                NOT_VALID = vpAvrg < 2
                # if preceding vehicle is all zero speed, we need different control
                if NOT_VALID and (T_BEG is not None):
                    ControlMode['StopLine'] = True
                    break

            # divide into intervals of 5 seconds
            tIntDur = 5 # second
            tIntBeg = np.arange(t[0],t[-1]-1,tIntDur)
            tIntEnd = np.arange(t[0]+tIntDur,t[-1]+1,tIntDur)  

            #  get all the tau points
            n = 3 # polynomial of order 5, for LGL need to minus 1
            #tauList = legendre_gauss_lobatto_nodes(n)
            tauList = np.array([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0])

            fArr = [np.nan]*len(tIntBeg)
            fBasis = [np.nan]*(len(tIntBeg))
            dpBasis = [np.nan]*(len(tIntBeg))
            for i, _ in enumerate(tIntBeg):
                tList = ((1-tauList)*tIntBeg[i] + (1+tauList)*tIntEnd[i])/2
                dpVal = np.interp(tList, t, dp)

                L, Lbasis = self.__GetLagrangeCoeff(n,tList,dpVal)

                fArr[i] = np.flip(L)
                fBasis[i] = np.array(Lbasis)
                dpBasis[i] = dpVal

            self.fArr = fArr # coefficients of lower order first, higher order last [p0,p1,p2,p3] -> p3*t^3+p2*t^2+p1*t+p0
            self.fBasis = fBasis
            self.dpBasis = dpBasis

            self.nPoly = n
            self.nIntvl = len(tIntBeg)
            self.nIntvlIdx = round(tIntDur/self.dt)

            self.dmax = 80
            self.dmin = 10

            if INIT_STATE is None:
                dfollow0 = np.random.uniform(self.dmin+5, self.dmax-5)
                # initial conditions
                d0 = dp[0]-dfollow0
                v0 = min(max(np.random.uniform(vp[0]-5, vp[0]+5), 0), self.vmax)
            else:
                d0=INIT_STATE['d0']
                v0=INIT_STATE['v0']
                #a0=INIT_STATE['a0']
            # terminal conditions
            df = dp[-1]-30 #dp[-1]-dfollow0
            vf = vp[-1]
            af = 0

        self.SELECT_PREC_ID = SELECT_PREC_ID
        self.vehId = vehId
        
        N = len(t)-1
        self.N = N

        self.t = t

        self.dp = dp
        self.vp = vp
        self.ap = ap

        self.d0 = d0
        self.v0 = v0
        #self.a0 = a0
        self.df = df
        self.vf = vf
        self.af = af

        self.tBeg = tBeg
        self.tBegIdx = tBegIdx

        if tEnd >= time[-1]-self.dt:
            ControlMode['Terminate'] = True

        Info = {
                'fArr': fArr,
                'fBasis': fBasis,
                'dpBasis': dpBasis,
                }
        
        self.ControlMode = ControlMode
        return ControlMode, Info

    def reset(self, options={}):
        _,_ = self.updatePrecedingVehicle(options=options)

        # internal state to track the current position and speed of this vehicle
        self.state = torch.FloatTensor([self.d0, self.v0])
        self.state_dim = len(self.state)
        self.info = {}
        self.k = 0
        
        self.smin = np.array([self.dp[0]-self.dmax-1, self.vmin])
        self.smax = np.array([self.dp[-1]-self.dmin+1, self.vmax])

        self.state_dim = len(self.state)

        self.observation = self.state2Observation(self.state)[0,:]
        self.obs_dim = len(self.observation)

        self.action_dim = 1
        
        if self.SELECT_OBSERVATION == 'none':
            # df, v, k
            self.xmean = torch.FloatTensor([50., 5., 50.])
            self.xstd = torch.FloatTensor([10., 5., 30])   
            
        elif self.SELECT_OBSERVATION == 'poly':
            self.xmean = torch.FloatTensor([50., 5., 50., -1, 1, 40, 40, 4, -14, 122, 122, -9, 17, 171, 176])
            self.xstd = torch.FloatTensor([10., 5., 30., 25, 32, 25, 25, 30, 75, 65, 52, 148, 434, 421, 297])
            self.obsmin = torch.FloatTensor([0., 0., 0., -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5, -1e5])
            self.obsmax = torch.FloatTensor([self.dmax+5, self.vmax+1., self.N+2, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5])
            
        
        self.umean = torch.FloatTensor([0])
        self.ustd = torch.FloatTensor([1])

        return self.observation, None

    def getNextState(self, state, action):
        # make sure np array is N-by-1
        state = state.reshape(-1,self.state_dim)
        action = action.reshape(-1,1)

        # np array input
        d = state[:,0]
        v = state[:,1]
        a = action[:,0]

        if state.shape[0] != action.shape[0]:
            stateNext = torch.empty((action.shape[0], state.shape[1]))
        else:
            stateNext = torch.empty(state.shape)

        # move forward one step to get next state
        stateNext[:,0] = d + self.dt*v
        stateNext[:,1] = torch.minimum(torch.maximum(v + self.dt*a, torch.tensor(self.vmin)), torch.tensor(self.vmax))
        return stateNext
    

    def state2Observation(self, state, k=None, vehId=None, tBeg=None):

        # make sure np array is N-by-1
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)
        state = state.reshape(-1,self.state_dim)

        # handle different k input
        if k is None:
            k = self.k
        if not torch.is_tensor(k):
            k = k*torch.ones(state.shape[0]).int()

        if state.shape[0] != k.shape[0]:
            k = k[0]*torch.ones(state.shape[0]).int()
        if vehId is not None:
            if state.shape[0] != vehId.shape[0]:
                vehId = vehId[0]*torch.ones(state.shape[0]).int()
            if state.shape[0] != tBeg.shape[0]:
                tBeg = tBeg[0]*torch.ones(state.shape[0]).int()

        #dpt, vpt = self.getPrecedingVehicle(k, vehId, tBeg)
        kClip = torch.minimum(k, torch.FloatTensor([self.N])).int()
        dpt = torch.FloatTensor(self.dp)[kClip]
        vpt = torch.FloatTensor(self.vp)[kClip]
        if self.SELECT_OBSERVATION == 'none':
            observation = torch.column_stack((state[:,0],
                                state[:,1]))

        elif self.SELECT_OBSERVATION == 'poly':
            # idx = torch.minimum(torch.floor(k/self.nIntvlIdx),torch.tensor(self.nIntvl-1)).int()
            
            # dpNext = torch.matmul(torch.FloatTensor([[np.power(self.dt,3), np.power(self.dt,2), np.power(self.dt,1), 1]]), \
                                        # torch.transpose(torch.FloatTensor(self.fArr)[idx],0,-1)*torch.row_stack([(k+1)**3, (k+1)**2, k+1, torch.ones(k.shape)]))
            observation = torch.empty(state.shape[0],self.state_dim+1+self.nIntvl*(self.nPoly+1))
            observation[:,0] = state[:,0]
            observation[:,1] = state[:,1]
            observation[:,2] = k
            
            # observation poly, from highest order to zero order
            # fArr [p0,p1,p2,p3] -> p3*t^3+p2*t^2+p1*t+p0
            for i in range(self.nIntvl):
                for j in range(self.nPoly):
                    observation[:,self.state_dim+1+i*(self.nPoly+1)+j] = self.fArr[i][self.nPoly-j]*np.power(self.dt*(k+1), self.nPoly-j)
                observation[:,self.state_dim+1+i*(self.nPoly+1)+self.nPoly] = observation[:,self.state_dim+1+i*(self.nPoly+1)+self.nPoly-1] + self.fArr[i][0]
                pass
            pass
        elif self.SELECT_OBSERVATION == 'all':
            observation = torch.empty(state.shape[0],self.state_dim+1+self.N)
            observation[:,0:self.state_dim] = state
            observation[:,self.state_dim] = k
            observation[:,self.state_dim+1:self.state_dim+1+self.N] = torch.tensor(self.dp[:self.N])
        return observation

    def step(self, action):
        info = {}
        # clip action
        # dmin = self.dlbFunc(self.vfinal)
        obs = self.observation.reshape((-1,self.obs_dim))


        # state = self.state
        stateNext = self.getNextState(self.state, action)
        self.state = stateNext[0,:]

        reward = self.getReward(self.observation, action)
        # terminated = (self.dp[-1]-df>=self.df-dftol) & (self.dp[-1]-df<=self.df+dftol) & (v>=self.vf-vftol) & (v<=self.vf+vftol) & (k == self.N)
        # k = obs[:,2]
        
        truncated = (self.k == self.N)
        terminated = truncated
        observationNext = self.calcDyn(self.observation, action, IS_OBS=True)
        
        self.k = self.k + 1
        #observationNext = self.state2Observation(self.state)
        self.observation = observationNext[0,:]
        return observationNext[0,:], reward[0], terminated, truncated, None

    
    def replayEpisode(self, batch, PrecInfo=None):
        # PrecInfo: t, dp, vp  wq ewq e qw eqw eqw ewq

        observationBatch = torch.FloatTensor(batch[0])
        actionBatch = torch.FloatTensor(batch[2]).reshape(-1,self.action_dim)

        if PrecInfo is None:
            PrecInfo = {'t': self.t,
                        'd': self.dp,
                        'v': self.vp}
        #     df_final = self.df_final
            # vfinal = self.vfinal
        # else:
        df_final, vfinal = self.getDesiredFinalStates(observationBatch[-1,:].reshape(1,-1), observationBatch[-1,2])
        df_final = df_final.numpy()
        vfinal = vfinal.numpy()
            
        stateBatch = observationBatch[:,0:self.state_dim]
        
        TrajDict = {'d':{'follow': PrecInfo['d']-stateBatch[:,0], 'ubnd': np.hstack((self.dmax*np.ones(len(PrecInfo['t'])-1),df_final)), 'lbnd': np.hstack((self.dlbFunc(PrecInfo['v'][:-1]),df_final))},
                    'v': {'p_{}'.format(self.vehId): PrecInfo['v'], 'opt': stateBatch[:,1], 'ubnd': np.hstack((self.vmax*np.ones(len(PrecInfo['t'])-1),vfinal)), 'lbnd': np.hstack((self.vmin*np.ones(len(PrecInfo['t'])-1),vfinal))}, 
                    'a': {'opt': batch[2]}}
        
        xaxis = PrecInfo['t']

        return xaxis, TrajDict
        
    
    def __dpFunc(self, k):
        Tdict = self.Utils_c.checkDtype(k)
 
        return Tdict['func']['interp1'](np.arange(self.N+1), self.dp.flatten(), k.flatten())
    

    def calcDiff(self,  obs, act, obsnext, dAgent_dict, USE_CUDA=True):
        # used to do auto-differentiation
        pErr,uLoss,Info = self.Utils_c.calcDiff(obs, act, obsnext, dAgent_dict, self.dEnvDiff_dict, USE_CUDA=USE_CUDA)

        return pErr, uLoss, Info


    def __getBasisVal(self, fVal, k, i=0):
        Tdict = self.Utils_c.checkDtype(k)
        kCalc = Tdict['func']['clip'](k, i*self.nIntvlIdx, (i+1)*self.nIntvlIdx)
    
        basisVal = np.ones(kCalc.shape)*np.sum(fVal.reshape(-1,1)*np.array([(self.dt*(kCalc+1))**3, (self.dt*(kCalc+1))**2, (self.dt*(kCalc+1)), np.ones(kCalc.shape)]).reshape(len(fVal),-1))
        return basisVal
    
    def getDesiredFinalStates(self, obs, k):
        if self.SELECT_OBSERVATION == 'poly':     
            kCalc = k.reshape((-1,1)) # this is m-by-1 vector that used for calculation
            dpN = 0	# dp(N)
            dpN1 = 0 # dp(N-1)
            for j in range(self.nPoly+1):
                # skip linear term
                if j == self.nPoly-1:
                    continue
                dpN = dpN + obs[:,self.state_dim+(self.nIntvl-1)*(self.nPoly+1)+j].reshape((-1,1))
                # if not last
                if j != self.nPoly:
                    dpN1 = dpN1 + obs[:,self.state_dim+(self.nIntvl-1)*(self.nPoly+1)+j].reshape((-1,1))/(kCalc+1)**(self.nPoly-j)*(kCalc)**(self.nPoly-j)
                else:
                    dpN1 = dpN1 + obs[:,self.state_dim+(self.nIntvl-1)*(self.nPoly+1)+j].reshape((-1,1))-(obs[:,self.state_dim+(self.nIntvl-1)*(self.nPoly+1)+j-1].reshape((-1,1))/(kCalc+1))
        else:
            #vfinal = self.vp[-1]
            #vfinal = 15.08100945 #((self.dp[-1]-self.dp[-2])/self.dt)
            dpN = 0	# dp(N)
            dpN1 = 0 # dp(N-1)
            for j in range(self.nPoly+1):
                dpN = dpN + self.fArr[self.nIntvl-1][self.nPoly-j]*np.power(self.dt*(kCalc+1), self.nPoly-j)
                dpN1 = dpN1 + self.fArr[self.nIntvl-1][self.nPoly-j]*np.power(self.dt*(kCalc), self.nPoly-j)
            pass    

        #yvfinal = self.vp[-1]
        vfinal = ((dpN-dpN1)/self.dt)[:,0]    
        df_final = 1+2.5*vfinal
    
        return df_final, vfinal

    def getTerminalReward(self, xVar, action):
        
        if self.k == self.N:
            action = action.reshape((-1,self.action_dim))
            obs = xVar.reshape((-1,self.obs_dim))
            d = obs[:,0]
            v = obs[:,1]
            
            df = self.dp[-1]-d
            vf = self.vp[-1]
            
            dmin = self.ht*vf + self.dmin
            reward = self.w5*(df-dmin)**2 +self.w6*(vf-v)**2
        else:
            reward = 0
        return reward 
        
    
    def getReward(self, xVar, action, IS_OBS=True):
        action = action.reshape((-1,self.action_dim))
        a = action[:,0]
       
        obs = xVar.reshape((-1,self.obs_dim))

        d = obs[:,0]
        v = obs[:,1]
        k = self.k

        p1 = self.Veh['p1']
        p2 = self.Veh['p2']
        p3 = self.Veh['p3']
        # penalize k==self.N, which is the last state
        # df_final, vfinal = self.getDesiredFinalStates(obs, k)
        dp = self.dp[self.k]
        vp = self.vp[self.k]
        df = dp-d 
        
        df_upper_penalty = max(df-self.dmax,0)
        df_lower_penalty = max(self.dmin-df,0)
        reward = 0
        if self.k == self.N:
            reward = self.getTerminalReward(xVar, action)
        else:
            pow=(p1*v+p2*(v**3)+p3*(v*a))
            reward +=  self.w1*(a**2) + self.w2*(pow)
            reward += self.w3*(df_upper_penalty**2) + self.w4*(df_lower_penalty**2)
        reward = -reward*0.01
        return torch.tensor([reward]) 

    def calcDyn(self, xVar, action, IS_OBS=True):
        action = action.reshape((-1,self.action_dim))
        a = action[:,0].reshape((-1,1))
        k = torch.tensor(self.k).reshape((-1,1))
        
        if not IS_OBS:
            xVar = xVar.reshape((-1,self.state_dim))
            d = xVar[:,0].reshape((-1,1))
            v = xVar[:,1].reshape((-1,1))     

            dyn = torch.hstack((d+self.dt*v,
                                    torch.clip(v+self.dt*a, self.vmin, self.vmax)))
        else:        
            obs = xVar.reshape((-1,self.obs_dim))

            d = obs[:,0].reshape((-1,1))
            v = obs[:,1].reshape((-1,1))
            if self.SELECT_OBSERVATION == 'none':      
                dyn = torch.hstack([(d+self.dt*v),
                                    torch.clip(v+self.dt*a, self.vmin, self.vmax)])
            elif self.SELECT_OBSERVATION == 'poly':                
                def __dpFunc(obs, k):      
                    # use sigmoid to create lookup table based on k
                    dpCalc = 0
                    for i in range(self.nIntvl):
                        for j in range(self.nPoly+1):
                            # skip linear term
                            if j == self.nPoly-1:
                                continue
                            # if last interval
                            if i == self.nIntvl-1:
                                dpCalc = dpCalc + obs[:,3+i*(self.nPoly+1)+j].reshape((-1,1))*(torch.sigmoid((k+0.5-i*self.nIntvlIdx)))
                            else:
                                dpCalc = dpCalc + obs[:,3+i*(self.nPoly+1)+j].reshape((-1,1))*(torch.sigmoid((k+0.5-i*self.nIntvlIdx))-torch.sigmoid((k+0.5-(i+1)*self.nIntvlIdx)))
                    return dpCalc
                
                dpDelta = -__dpFunc(obs,k)
                dyn = torch.hstack([(d + self.dt*v),
                                    torch.clip(v+self.dt*a, self.vmin, self.vmax),
                                    k])
                for i in range(self.nIntvl):
                    for j in range(self.nPoly+1):
                        # if not last
                        if j != self.nPoly:
                            dyn = torch.hstack([dyn,
                                        obs[:,self.state_dim+i*(self.nPoly+1)+j].reshape((-1,1))*(k+2)**(self.nPoly-j)/(k+1)**(self.nPoly-j)])
                        else:
                            dyn = torch.hstack([dyn,
                                        obs[:,self.state_dim+i*(self.nPoly+1)+j].reshape((-1,1)) + obs[:,self.state_dim+i*(self.nPoly+1)+j-1].reshape((-1,1))*((k+2)/(k+1)-1)])

            elif self.SELECT_OBSERVATION == 'all':
                dyn = xVar
                dyn[0] = d + self.dt*v
                dyn[1] = torch.clip(v+self.dt*a, self.vmin, self.vmax)
                dyn[2] = k + 1 
                dyn= dyn.reshape((-1,self.obs_dim))
        return dyn