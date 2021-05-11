import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class PPONetwork(nn.Module):
    def __init__(self,state_dim,action_dim,n_conv=128,n_fc=128,n_fc1=128):
        super(PPONetwork,self).__init__()
        self.s_dim=state_dim
        self.a_dim=action_dim
        self.vectorOutDim=n_conv
        self.scalarOutDim=n_fc
        self.numFcInput=2 * self.vectorOutDim * (self.s_dim[1]-4+1) + 3 * self.scalarOutDim + self.vectorOutDim*(self.a_dim-4+1)
        self.numFcOutput=n_fc1

        #-------------------define layer-------------------
        self.tConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.dConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.cConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.bufferFc=nn.Linear(1,self.scalarOutDim)

        self.leftChunkFc=nn.Linear(1,self.scalarOutDim)

        self.bitrateFc=nn.Linear(1,self.scalarOutDim)


        self.pi_fc=nn.Linear(self.numFcInput,self.numFcOutput)
        
        self.value_fc=nn.Linear(self.numFcInput,self.numFcOutput)

        self.pi_net=nn.Linear(self.numFcOutput,self.a_dim)

        self.value_net=nn.Linear(self.numFcOutput,1)
        #------------------init layer weight--------------------
        # tensorflow-1.12 uses glorot_uniform(also called xavier_uniform) to initialize weight
        # uses zero to initialize bias
        # Conv1d also use same initialize method 
        nn.init.xavier_uniform_(self.bufferFc.weight.data)
        nn.init.constant_(self.bufferFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.leftChunkFc.weight.data)
        nn.init.constant_(self.leftChunkFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.bitrateFc.weight.data)
        nn.init.constant_(self.bitrateFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.tConv1d.weight.data)
        nn.init.constant_(self.tConv1d.bias.data,0.0)
        nn.init.xavier_uniform_(self.dConv1d.weight.data)
        nn.init.constant_(self.dConv1d.bias.data,0.0)
        nn.init.xavier_normal_(self.cConv1d.weight.data)
        nn.init.constant_(self.cConv1d.bias.data,0.0)
        nn.init.xavier_normal_(self.pi_fc.weight.data)
        nn.init.constant_(self.pi_fc.bias.data,0.0)
        nn.init.xavier_normal_(self.value_fc.weight.data)
        nn.init.constant_(self.value_fc.bias.data,0.0)
        nn.init.xavier_normal_(self.pi_net.weight.data)
        nn.init.constant_(self.pi_net.bias.data,0.0)
        nn.init.xavier_normal_(self.value_net.weight.data)
        nn.init.constant_(self.value_net.bias.data,0.0)


    def forward(self,inputs):

        bitrateFcOut=F.relu(self.bitrateFc(inputs[:,0:1,-1]),inplace=True)

        bufferFcOut=F.relu(self.bufferFc(inputs[:,1:2,-1]),inplace=True)
 
        tConv1dOut=F.relu(self.tConv1d(inputs[:,2:3,:]),inplace=True)

        dConv1dOut=F.relu(self.dConv1d(inputs[:,3:4,:]),inplace=True)

        cConv1dOut=F.relu(self.cConv1d(inputs[:,4:5,:self.a_dim]),inplace=True)
       
        leftChunkFcOut=F.relu(self.leftChunkFc(inputs[:,5:6,-1]),inplace=True)

        t_flatten=tConv1dOut.view(tConv1dOut.shape[0],-1)

        d_flatten=dConv1dOut.view(dConv1dOut.shape[0],-1)

        c_flatten=cConv1dOut.view(dConv1dOut.shape[0],-1)

        fullyConnectedInput=torch.cat([bitrateFcOut,bufferFcOut,t_flatten,d_flatten,c_flatten,leftChunkFcOut],1)

        pi_fc_out=F.relu(self.pi_fc(fullyConnectedInput),inplace=True)

        value_fc_out=F.relu(self.value_fc(fullyConnectedInput),inplace=True)

        pi_out=F.softmax(self.pi_net(pi_fc_out))

        value_out=self.value_net(value_fc_out)

        return pi_out, value_out


