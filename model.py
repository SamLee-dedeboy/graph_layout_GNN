from cmath import atanh
from email.mime import base
from sklearn import preprocessing
import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import sys
from torch_geometric.datasets import QM9
import networkx as nx
import preprocess
import MDS
import numpy as np
import torch_geometric.utils.convert as cv
import netrd
import matplotlib.pyplot as plt
import os
import layout_metrics
sys.path.insert(0, 'geo_gcn/src')
#from geo_gcn.src.architectures import SGCN
#from geo_gcn.src.graph_conv import SpatialGraphConv
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
# Defines the architecture of VGAE. 
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Defines the architecture of GAE
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
#
# Failed attempts for using SGCN. Will not be used in the project
#
# class SGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = SpatialGraphConv(2, in_channels, 2 * out_channels, 64)
#         self.conv2 = SpatialGraphConv(2, 2 * out_channels, out_channels, 64)
#         #self.conv_mu = SpatialGraphConv(2, in_channels, out_channels, 64)
#         #self.conv_logstd = SpatialGraphConv(2, in_channels, out_channels, 64)

#     def forward(self, x, pos, edge_index):
#         x = self.conv1(x, pos, edge_index).relu()
#         return self.conv2(x, pos, edge_index)
#         # res1 = self.conv_mu(data.x, data.pos, data.edge_index)
#         # res2 = self.conv_logstd(data.x,data.pos, data.edge_index)
#         # return res1, res2
# class VariationalSGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = SpatialGraphConv(2, in_channels, 2 * out_channels, 64)
#         self.conv_mu = SpatialGraphConv(2, 2 * out_channels, out_channels, 64)
#         self.conv_logstd = SpatialGraphConv(2, 2 * out_channels, out_channels, 64)

#     def forward(self, x, pos, edge_index):
#         x = self.conv1(x, pos, edge_index).relu()
#         return self.conv_mu(x, pos, edge_index), self.conv_logstd(x, pos, edge_index)
 
#         # res1 = self.conv_mu(data.x, data.pos, data.edge_index)
#         # res2 = self.conv_logstd(data.x,data.pos, data.edge_index)
#         # return res1, res2
#
# -------------------------------
#

# Wrapper class for the project, defines model, layouts & stuff
class LayoutRater:
    def __init__(self):
        self.variational_flag = False
        self.num_test_layout = 4
        self.in_channels = 2
        self.out_channels = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # loads data from 'train' or 'test'.
    # The files are preprocessed in preprocess.py
    def loadData(self, train_filename, test_filename):
        dirname = os.path.dirname(__file__)

        train_filepath = os.path.join(dirname, train_filename) 
        test_filepath = os.path.join(dirname, test_filename) 
        train_data = torch.load(train_filepath)
        test_data = torch.load(test_filepath)
        return train_data, test_data
    
    # loads trained model to avoid retraining every run.
    # model_name should be either "VGAE" or "GAE"
    # assumes that the model is located in same folder with filename "VGAE" or "GAE"
    def load_model(self, model_name):
        if model_name == "VGAE":
                model = VGAE(VariationalGCNEncoder(self.in_channels, self.out_channels))
        else:
                model = GAE(GCNEncoder(self.in_channels, self.out_channels))
        dirname = os.path.dirname(__file__)
        model_filepath = os.path.join(dirname, model_name)
        model.load_state_dict(torch.load(model_filepath))
        model=model.to(self.device)
        return model
    
    # train & save model
    # train_loader: train dataset
    # test_loader: test dataset.
    # assumed to be of type DataLoader in torch_geometric.loader
    def generate_model(self, train_loader, test_loader):
        #graphList = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
        # initializing model
        in_channels, out_channels = self.in_channels, self.out_channels
        if self.variational_flag:
                model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
                model_name="VGAE"
        else:
                model = GAE(GCNEncoder(in_channels, out_channels))
                model_name="GAE"
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # train
        epochs = 3
        train_loss_values=[]
        val_loss_values=[]
        for epoch in range(epochs):
            train_loss = self.train(model=model, optimizer=optimizer, train_loader=train_loader)
            train_loss_values.append(train_loss.item())
            
            # use test dataset also as validation dataset for preliminary experiments
            #val_loss = self.validate(model=model, dataLoader=test_loader)
            #val_loss_values.append(val_loss.item())

            print(epoch, train_loss.item())
            print("-----------")

        # save model
        dirname = os.path.dirname(__file__)
        model_filepath = os.path.join(dirname, model_name)
        torch.save(model.state_dict(), model_filepath)

        #
        # plot the loss curve of train dataset and validation dataset
        # used in preliminary experiments.
        #
        # fig=plt.figure(1, figsize=(8, 6), dpi=80)
        # ax = fig.add_subplot(111)
        # ax.plot(train_loss_values, c='red')
        # ax.plot(val_loss_values, c='blue')
        # plt.show() 

    # train the model
    # uses different loss functions for GAE & VGAE
    def train(self, model, optimizer, train_loader):
        model.train()
        total_loss = 0;
        for train_data in train_loader:
            train_data = train_data.to(self.device)
            optimizer.zero_grad()
            z = model.encode(train_data.x, train_data.edge_index)
            #print(z)
            loss = model.recon_loss(z, train_data.pos_edge_index)
            #print(loss)
            if self.variational_flag:
                #loss = model.kl_loss()
                num_nodes=train_data.num_nodes
                loss = loss + (1 / num_nodes) * model.kl_loss()
            #print(index, "train loss:", loss.item())
            total_loss += loss
            loss.backward()
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    # Valiation function used in preliminary experiments
    # *can ignore
    @torch.no_grad()
    def validate(self, model, dataLoader):
        model.eval()
        from collections import defaultdict
        graph_loss_dict = defaultdict(lambda: [0] * self.num_test_layout)
        total_loss=0
        for data in dataLoader:
            #print(data.graph_tag.item(), data.layout_tag.item())
            baseline_graph = self.recon_baseline_graph(data)
            #print(data.layout_tag)
            data = data.to(self.device)
            z = model.encode(data.x,  data.edge_index)
            
            auc, ap = model.test(z, data.pos_edge_index, data.neg_edge_index)
            print(data.graph_tag.item(), data.layout_tag.item(), f'AP: {ap:.4f}')
            #print(z.shape)
            #print(model.decode(z, data.pos_edge_index).shape)
            recon_graph = preprocess.edge_index_to_graph(data.num_nodes, data.pos_edge_index.cpu().numpy(), model.decode(z, data.pos_edge_index))
            assert nx.number_of_nodes(recon_graph) == data.num_nodes
            tmp_data = Data(x=data.x, edge_index=data.pos_edge_index)
            original_graph = cv.to_networkx(tmp_data, to_undirected=True)
            #print(original_graph)
            dist = L1_dist(recon_graph, original_graph)
            #base_dist = L1_dist(baseline_graph, original_graph)
            loss = model.recon_loss(z, data.pos_edge_index)
            if self.variational_flag:
                loss = loss + (1 / data.num_nodes) * model.kl_loss()
            #loss = model.kl_loss()
            total_loss+=loss
            #print(data.graph_tag.item(), data.layout_tag.item(), "loss=", "{0:.5f}".format(loss.item()), "dist=", "{0:.5f}".format(dist))
            graph_loss_dict[data.graph_tag.item()][data.layout_tag.item()] = loss.item()
            
        #preprocess.showLayout(dataLoader)
        return total_loss/len(dataLoader)

    # test the model
    def test(self, model, dataLoader):
        model.eval()
        total_loss=0
    
        #test_data_with_metric = []
        losses = []
        for data in dataLoader:
            data = data.to(self.device)
            z = model.encode(data.x,  data.edge_index)
            
            
            # get link prediction results. 
            #auc, ap = model.test(z, data.pos_edge_index, data.neg_edge_index)

            # Get reconstructed grap with type networkx.graph
            recon_graph = preprocess.edge_index_to_graph(data.num_nodes, data.pos_edge_index.cpu().numpy(), model.decode(z, data.pos_edge_index))
            assert nx.number_of_nodes(recon_graph) == data.num_nodes
            # Get original graph with type networkx.graph
            tmp_data = Data(x=data.x, edge_index=data.pos_edge_index)
            original_graph = cv.to_networkx(tmp_data, to_undirected=True)

            # calculate L1 difference between reconstructed graph and original graph
            dist = L1_dist(recon_graph, original_graph)
            
            # calculate loss
            loss = model.recon_loss(z, data.pos_edge_index)
            if self.variational_flag:
                loss = loss + (1 / data.num_nodes) * model.kl_loss()
            total_loss+=loss
            #
            # get aeathetic metrics on test graphs
            # E_c should take longer time to compute
            # if loading from 'test_data_w_metric_n', then avoid calculating metrics to save time
            # if loading from 'test', then need to uncomment the codes below for computing pearson_coefficient_correlation
            #
            
            # M_l = layout_metrics.edge_length_variation(data.cpu())
            # M_a = layout_metrics.minimum_angle(data.cpu())
            # print("calculating E_c...")
            # E_c = layout_metrics.edge_crossings(data.cpu())

            # guard
            if not hasattr(data, 'M_l'):
               data = preprocess.add_readability_metrics(data.cpu())
            
            M_l = data.M_l.item()
            M_a = data.M_a.item()
            E_c = data.E_c.item()

            #test_data_with_metric.append(data)

            #tag = data.graph_tag # for train_data 
            tag = data.graph_tag.item() # for test_data
            print(tag, data.layout_tag.item(), "loss=", "{0:.5f}".format(loss.item()), "dist=", "{0:.5f}".format(dist), "M_l=", "{0:.5f}".format(M_l), "M_a=", "{0:.5f}".format(M_a), "E_c=", "{0:.5f}".format(E_c))
            
            losses.append(loss.item())

        # calculate pearson correlation between loss and aesthetic metrics
        pearson_correlation(losses=losses, dataLoader=dataLoader)
        return total_loss/len(dataLoader)

    # deprecated. used in preliminary experiments
    def recon_baseline_graph(self, data):
        #TS = data.x.numpy()
        TS = np.array(preprocess.generateNodeDist(data.x.cpu().numpy()))
        reconstructor = netrd.reconstruction.FreeEnergyMinimization()
        G = reconstructor.fit(TS, avg_k = 8.85)
        return G


def L1_dist(G1, G2):
    return np.absolute(nx.adjacency_matrix(G1).todense() - nx.adjacency_matrix(G2).todense()).sum()


# computes pearson coefficient correlation.
# if metrics is None, then assumes that dataLoader has attributes M_l, M_a and E_c
def pearson_correlation(losses, dataLoader, metrics=None):
    if metrics == None:
        M_l_samples = np.array([data.M_l.item() for data in dataLoader])
        M_a_samples = np.array([data.M_a.item() for data in dataLoader])
        E_c_samples = np.array([data.E_c.item() for data in dataLoader])
    else:
        M_l_samples = np.array([data.M_l for data in metrics])
        M_a_samples = np.array([data.M_a for data in metrics])
        E_c_samples = np.array([data.E_c for data in metrics])
    losses = np.array(losses)
    
    corr_M_l = np.corrcoef(losses, M_l_samples)
    corr_M_a = np.corrcoef(losses, M_a_samples)
    corr_E_c = np.corrcoef(losses, E_c_samples)
    print(corr_M_l[0][1])
    print(corr_M_a[0][1])
    print(corr_E_c[0][1])


if __name__ == "__main__":
    # init model
    l = LayoutRater()
    # variational_flag is used to choose between GAE and VGAE.
    # True -> use VGAE
    # False -> use GAE
    l.variational_flag=False
    # load data from files.
    train_data, test_data = l.loadData('train', "test_data_w_metrics_0")

    # wrap in DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # train model 
    #l.generate_model(train_loader, test_loader)
    
    # load previously trained model
    # assumes that saved model is located in same folder and named 'VGAE' or 'GAE'
    if l.variational_flag:
        model = l.load_model('VGAE')
        print("VGAE")
    else:
        model = l.load_model('GAE')
        print("GAE")
    # test model
    l.test(model=model,dataLoader=test_loader)







