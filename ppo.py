import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network import PPONetwork
from datetime import datetime
from torch.distributions import Categorical


class PPO(object):
    def __init__(self,s_dim,action_dim,actor_lr=1e-4):
        self.s_dim=s_dim
        self.a_dim=action_dim
        self.discount=0.99
        self.entropy_eps=1e-6
        self.action_eps=1e-4
        self._entropy=0.5
        self.clip=0.2 
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actorNetwork=PPONetwork(self.s_dim,self.a_dim).double().to(self.device)
        self.actorOptim=torch.optim.Adam(self.actorNetwork.parameters(),lr=actor_lr)
        self.actorOptim.zero_grad()

        self.loss_function=nn.MSELoss()

    def set_entropy_decay(self, decay=0.6):
        self._entropy *= decay
 

    def get_entropy(self):
        return np.clip(self._entropy, 0.01, 5.)

    def getNetworkGradient(self,s_batch,a_batch,R_batch,old_pi_batch):
        self.actorOptim.zero_grad()

        s_batch=torch.from_numpy(s_batch).to(self.device)
        a_batch=torch.from_numpy(a_batch).to(self.device)
        R_batch=torch.from_numpy(R_batch).to(self.device)
        old_pi_batch=torch.from_numpy(old_pi_batch).to(self.device)


        probability,values=self.actorNetwork.forward(s_batch)
        probability=probability.clamp(self.action_eps,1-self.action_eps)
        log_prob=torch.log(torch.sum(probability*a_batch,1,keepdim=True))

        entropy=probability*torch.log(probability)

        adv=(R_batch-values).detach()

        probs=torch.sum(probability*a_batch,1,keepdim=True)

        old_probs=torch.sum(old_pi_batch*a_batch,1,keepdim=True)

        ratio=probs/old_probs

        surr1=ratio*adv
        surr2=ratio.clamp(1-self.clip,1+self.clip)*adv
        ppo2loss=torch.min(surr1,surr2)

        dual_loss=((adv<0).type(torch.float64))*torch.max(ppo2loss,3.0*adv)+((adv>=0).type(torch.float64))*ppo2loss

        loss=-torch.sum(dual_loss)+self.get_entropy()*torch.sum(entropy)

        loss.backward(retain_graph=True)

        val_loss=self.loss_function(R_batch,values)
        val_loss.backward()

        self.actorOptim.step()

    def actionSelect(self,stateInputs):
        stateInputs=torch.from_numpy(stateInputs).to(self.device)
        with torch.no_grad():
            probability,value=self.actorNetwork.forward(stateInputs)
            probability=probability.clamp(self.action_eps,1-self.action_eps)
            return probability.cpu().numpy()


    def compute_v(self,s_batch,a_batch,r_batch,terminal):
        s_batch=torch.from_numpy(np.stack(s_batch,axis=0)).to(self.device)
        r_batch=torch.from_numpy(np.vstack(r_batch)).to(self.device)
        R_batch=torch.zeros(r_batch.shape,dtype=torch.double).to(self.device)

        if terminal:
            R_batch[-1,0]=0
        else:
            with torch.no_grad():
                _,v_batch=self.actorNetwork.forward(s_batch).to(self.device)
            R_batch[-1,0]=v_batch[-1,0]
        for t in reversed(range(r_batch.shape[0]-1)):
            R_batch[t,-1]=r_batch[t,-1] + self.discount*R_batch[t+1,-1]
        return list(R_batch)



    def hardUpdateActorNetwork(self,actor_net_params):
        for target_param,source_param in zip(self.actorNetwork.parameters(),actor_net_params):
            target_param.data.copy_(source_param.data)
 
    def updateNetwork(self):
        # use the feature of accumulating gradient in pytorch
        self.actorOptim.step()
        self.actorOptim.zero_grad()
    def getActorParam(self):
        return list(self.actorNetwork.parameters())


if __name__ =='__main__':
    SINGLE_S_LEN=19

    AGENT_NUM=5
    BATCH_SIZE=200

    S_INFO=6
    S_LEN=8
    ACTION_DIM=6

    discount=0.9



    obj=PPO([S_INFO,S_LEN],ACTION_DIM)
    timenow=datetime.now()

    episode=3
    for i in range(episode):

        state2Select=np.random.randn(1,S_INFO,S_LEN)
        state=np.random.randn(AGENT_NUM,S_INFO,S_LEN)
        old_po=np.random.randn(AGENT_NUM,ACTION_DIM)
        action=np.random.randint(0,3,(AGENT_NUM,1))
        reward=np.random.randn(AGENT_NUM,1)
        probability=obj.actionSelect(state2Select)
        obj.getNetworkGradient(state,action,reward,old_po)


    print('train:'+str(episode)+' times use:'+str(datetime.now()-timenow))



