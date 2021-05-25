import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.utils.data.sampler import SubsetRandomSampler
from topo_data import Autopo, split_balance_data
import numpy as np
import math
import csv
from scipy import stats
from easydict import EasyDict
import argparse

from model import CircuitGNN, PT_GNN, MT_GNN
import copy

def rse(y,yt):

    assert(y.shape==yt.shape)

    var=0
    m_yt=yt.mean()
#    print(yt,m_yt)
    for i in range(len(yt)):
        var+=(yt[i]-m_yt)**2 

    mse=0
    for i in range(len(yt)):
        mse+=(y[i]-yt[i])**2

    rse=mse/(var+0.0000001)

    rmse=math.sqrt(mse/len(yt))

#    print(rmse)

    return rse


def initialize_model(model_index, gnn_nodes, gnn_layers, pred_nodes, nf_size, ef_size,device):
    args = EasyDict()
    args.len_hidden = gnn_nodes
    args.len_hidden_predictor = pred_nodes
    args.len_node_attr = nf_size
    args.len_edge_attr = ef_size
    args.gnn_layers = gnn_layers
    args.use_gpu = False
    args.dropout = 0.0

    if model_index==0:
        model = CircuitGNN(args).to(device)
        return model
    elif model_index==1:
        model = PT_GNN(args).to(device)
        return model
    elif model_index==2:
        model = MT_GNN(args).to(device)
        return model
    else:
        assert("Invalid model")

def train(train_loader, val_loader, model, n_epoch, batch_size, num_node, device, model_index,optimizer):

    train_perform=[]
    val_perform=[]
    
    loss=0

    min_val_loss=100

    for epoch in range(n_epoch):
    
    ########### Training #################
        
        train_loss=0
        n_batch_train=0
    
        model.train()
 

        for i, data in enumerate(train_loader):
                 data.to(device)
                 L=data.node_attr.shape[0]
                 B=int(L/num_node)
                 node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                 if model_index == 0:
                     edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                 else:
                     edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                     edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])
 
                 adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                 y=data.label
                       
                 n_batch_train=n_batch_train+1
                 optimizer.zero_grad()
                 if model_index == 0:
                      out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                 else:
                      out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device)))
 
                 out=out.reshape(y.shape)
                 assert(out.shape == y.shape)
                 loss=F.mse_loss(out, y.float())
                 loss.backward()
                 optimizer.step()
        
                 train_loss += out.shape[0] * loss.item()
        
        if epoch % 1 == 0:
                 print('%d epoch training loss: %.3f' % (epoch, train_loss/n_batch_train/batch_size))

                 n_batch_val=0
                 val_loss=0
 
                 model.eval()

                 for data in val_loader:

                     n_batch_val+=1

                     data.to(device)
                     L=data.node_attr.shape[0]
                     B=int(L/num_node)
                     node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                     if model_index == 0:
                         edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                     else:
                         edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                         edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])
     
                     adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                     y=data.label
                           
                     n_batch_train=n_batch_train+1
                     optimizer.zero_grad()
                     if model_index == 0:
                          out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                     else:
                          out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device)))
     
                     out=out.reshape(y.shape)
                     assert(out.shape == y.shape)
                     loss=F.mse_loss(out, y.float())
                     val_loss += out.shape[0] * loss.item()
                 val_loss_ave=val_loss/n_batch_val/batch_size

                 if val_loss_ave<min_val_loss:
                    model_copy=copy.deepcopy(model)
                    print(val_loss_ave,epoch)
                    epoch_min=epoch
                    min_val_loss=val_loss_ave
                 if epoch-epoch_min>3:
                    return model_copy                  
                     

        train_perform.append(train_loss/n_batch_train/batch_size)

    return model      


def test(test_loader, model, n_epoch, batch_size, num_node, model_index, flag,device, th):

        model.eval()        
        accuracy=0
        n_batch_test=0
        gold_list=[]
        out_list=[]
        analytic_list = []

        if flag==1:
            for data in test_loader:
                 data.to(device)
                 L=data.node_attr.shape[0]
                 B=int(L/num_node)
                 node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                 if model_index == 0:
                     edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                 else:
                     edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                     edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])
 
                 adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                 gold=data.label.cpu().detach().numpy()

                 n_batch_test=n_batch_test+1
                 if model_index==0:
                      out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
                 else:
                      out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
                 out=out.reshape(-1)
                 out=np.array([int(x>th) for x in out])
                 gold_list.extend(gold)
                 out_list.extend(out) 
                 gold=gold.reshape(-1)
                 np.set_printoptions(precision=2,suppress=True)
 
            true_positive=0
            true_negative=0
            false_negative=0
            false_positive=0
            
            for i in range(len(out_list)):
                if gold_list[i]==out_list[i]==1:
                    true_positive+=1
                if gold_list[i]==out_list[i]==0:
                    true_negative+=1
                if gold_list[i]!=out_list[i] and out_list[i]==0:
                    false_negative+=1
                if gold_list[i]!=out_list[i] and out_list[i]==1:
                    false_positive+=1
            #print("Average time:",(end-start)/n_batch_test/batch_size)
            
            myCsvRow=[name,th,true_positive,true_negative,false_negative,false_positive]
            print("1-Spec:",false_positive/(true_negative+false_positive))
            print("recal:",true_positive/(true_positve_false_negative))

        else:
            for data in test_loader:
                 data.to(device)
                 L=data.node_attr.shape[0]
                 B=int(L/num_node)
                 node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                 if model_index == 0:
                     edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                 else:
                     edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                     edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])
 
                 adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                 y=data.label.cpu().detach().numpy()

                 n_batch_test=n_batch_test+1
                 if model_index==0:
                      out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
                 else:
                      out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
                 out=out.reshape(y.shape)
                 assert(out.shape == y.shape)
                 out=np.array([x for x in out])
                 gold=np.array(y.reshape(-1))
                 gold=np.array([x for x in gold])

                 gold_list.extend(gold)
                 out_list.extend(out) 
     
                 L=len(gold)
    #             print(out,gold)
                 rse_result=rse(out,gold)
                 np.set_printoptions(precision=2,suppress=True)
#
            print("Final RSE:",rse(np.reshape(out_list,-1),np.reshape(gold_list,-1)))



def compute_smooth_reward(eff, vout, target_vout = .5):
    a = abs(target_vout) / 15

    eff[np.logical_or(eff > 1, eff < 0)] = 0.

    return eff * (1.1 ** (-((vout - target_vout) / a) ** 2))

def compute_piecewise_liear_reward(eff, vout):
    eff[np.logical_or(eff > 1, eff < 0)] = 0.
    eff[np.logical_or(vout < .35, vout > .65)] = 0.

    return eff

compute_reward = compute_piecewise_liear_reward

def optimize_reward(test_loader, eff_model, vout_model,
                    n_epoch, batch_size, num_node, model_index, flag,device, th):
    n_batch_test = 0

    sim_list = []
    analytic_list = []
    out_list = []

    sim_opts = []
    analytic_performs = []
    gnn_performs = []

    for data in test_loader:
        data.to(device)
        L = data.node_attr.shape[0]
        B = int(L / num_node)
        node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
        if model_index == 0:
            edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
        else:
            edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
            edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

        adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])

        sim_eff = data.sim_eff.cpu().detach().numpy()
        sim_vout = data.sim_vout.cpu().detach().numpy()

        analytic_eff = data.analytic_eff.cpu().detach().numpy()
        analytic_vout = data.analytic_vout.cpu().detach().numpy()

        n_batch_test = n_batch_test + 1
        if model_index == 0:
            eff = eff_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
        else:
            eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()

        eff = eff.squeeze(1)
        vout = vout.squeeze(1)

        sim_list.extend(compute_reward(sim_eff, sim_vout))
        analytic_list.extend(compute_reward(analytic_eff, analytic_vout))
        out_list.extend(compute_reward(eff, vout))

        # sim opt
        sim_opts.append(np.max(sim_list))

        # analytic
        analytic_opt = np.argmax(analytic_list)
        analytic_performs.append(sim_list[analytic_opt])

        # gnn
        gnn_opt = np.argmax(out_list)
        gnn_performs.append(sim_list[gnn_opt])

    np.set_printoptions(precision=2, suppress=True)

    print("GNN RSE:", rse(np.array(out_list), np.array(sim_list)))
    print("Analytic RSE:", rse(np.array(analytic_list), np.array(sim_list)))

    print('sim', sim_opts)
    print('analytic', analytic_performs)
    print('gnn', gnn_performs)

