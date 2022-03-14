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

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

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
class LayoutRater:
    def __init__(self):
        # self.train_graph_list = [686, 0, 414, 1684]
        # self.test_graph_list = [3980, 698]
        self.variational_flag = False
        self.num_test_layout = 4
        self.in_channels = 2
        self.out_channels = 16
        #self.use_spatial = False
        #self.graph_id=3980
        #self.num_of_train_layout = 50
        # self.test_layout_func_list = [MDS.mds_layout, nx.spiral_layout, nx.spring_layout, nx.circular_layout]
        # self.num_test_layout = len(self.test_layout_func_list)
        # self.num_graph = len(self.test_graph_list)
        # self.train_data, self.test_data = preprocess.readData(self.train_graph_list, self.test_graph_list, self.test_layout_func_list)
        
        #self.original_graph = preprocess.convertGraph(self.graph_id)
        #self.baseline_train_data, self.baseline_test_data = preprocess.readData(self.graph_id, self.num_of_train_layout, self.test_layout_func_list, baseline=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device=torch.device('cpu')
    def loadData(self, train_filename, test_filename):
        dirname = os.path.dirname(__file__)

        train_filepath = os.path.join(dirname, train_filename) 
        test_filepath = os.path.join(dirname, test_filename) 
        train_data = torch.load(train_filepath)
        test_data = torch.load(test_filepath)
        return train_data, test_data
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
            
            #val_loss = self.validate(model=model, dataLoader=test_loader)
            #val_loss_values.append(val_loss.item())

            print(epoch, train_loss.item())
            print("-----------")
        # save model
        dirname = os.path.dirname(__file__)
        model_filepath = os.path.join(dirname, model_name)
        torch.save(model.state_dict(), model_filepath)
        #self.test(model=model,dataLoader=test_loader)
        # fig=plt.figure(1, figsize=(8, 6), dpi=80)
        # ax = fig.add_subplot(111)
        # ax.plot(train_loss_values, c='red')
        # ax.plot(val_loss_values, c='blue')
        # plt.show() 
#print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')

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
            
        #preprocess.showLayout(dataLoader, self.test_layout_func_list)
        return total_loss/len(dataLoader)
    def test(self, model, dataLoader):
        model.eval()
        from collections import defaultdict
        graph_loss_dict = defaultdict(lambda: [0] * self.num_test_layout)
        total_loss=0
        
        
        #test_data_with_metric = []
        losses = []
        for data in dataLoader:
            data = data.to(self.device)
            z = model.encode(data.x,  data.edge_index)
            auc, ap = model.test(z, data.pos_edge_index, data.neg_edge_index)
            recon_graph = preprocess.edge_index_to_graph(data.num_nodes, data.pos_edge_index.cpu().numpy(), model.decode(z, data.pos_edge_index))
            assert nx.number_of_nodes(recon_graph) == data.num_nodes
            tmp_data = Data(x=data.x, edge_index=data.pos_edge_index)
            original_graph = cv.to_networkx(tmp_data, to_undirected=True)
            #print(recon_graph, original_graph, data)
            #print(original_graph)
            dist = L1_dist(recon_graph, original_graph)
            #base_dist = L1_dist(baseline_graph, original_graph)
            loss = model.recon_loss(z, data.pos_edge_index)
            if self.variational_flag:
                loss = loss + (1 / data.num_nodes) * model.kl_loss()
            #loss = model.kl_loss()
            total_loss+=loss
            # M_l = layout_metrics.edge_length_variation(data.cpu())
            # M_a = layout_metrics.minimum_angle(data.cpu())
            # print("calculating E_c...")
            # E_c = layout_metrics.edge_crossings(data.cpu())
            if not hasattr(data, 'M_l'):
               data = preprocess.add_readability_metrics(data.cpu())
            M_l = layout_metrics.edge_length_variation(data.cpu())
            M_a = data.M_a.item()
            E_c = data.E_c.item()
            # M_l = 0
            # M_a = 0
            # E_c = 0
            data.M_l = M_l
            # data.M_a = M_a
            # data.E_c = E_c
            #test_data_with_metric.append(data)

            #tag = data.graph_tag # for train_data 
            tag = data.graph_tag.item() # for test_data
            print(tag, data.layout_tag.item(), ap, "loss=", "{0:.5f}".format(loss.item()), "dist=", "{0:.5f}".format(dist), "M_l=", "{0:.5f}".format(M_l), "M_a=", "{0:.5f}".format(M_a), "E_c=", "{0:.5f}".format(E_c))
            losses.append(loss.item())
            #graph_loss_dict[data.graph_tag.item()][data.layout_tag.item()] = loss.item()
           
            #print(data)
        #dirname = os.path.dirname(__file__)
        #test_filepath = os.path.join(dirname, 'test_data_w_metrics_' + str(test_graph_id))
        #torch.save(test_data_with_metric, test_filepath)
        #preprocess.showLayout(dataLoader)

        pearson_correlation(losses=losses, dataLoader=dataLoader)
        return total_loss/len(dataLoader)
    def recon_baseline_graph(self, data):
        #TS = data.x.numpy()
        TS = np.array(preprocess.generateNodeDist(data.x.cpu().numpy()))
        reconstructor = netrd.reconstruction.FreeEnergyMinimization()
        G = reconstructor.fit(TS, avg_k = 8.85)
        #print(dist)
        #print(G)
        return G
def L1_dist(G1, G2):
    return np.absolute(nx.adjacency_matrix(G1).todense() - nx.adjacency_matrix(G2).todense()).sum()


def compare(item1, item2):
    key1 = item1.split(" ")[0][2:-2]
    key2 = item2.split(" ")[0][2:-2]
    if key1 == key2:
        key11 = item1.split(" ")[1][0]
        key22 = item2.split(" ")[1][0]
        if key11 < key22:
            return -1
        else:
            return 1
    if key1 < key2:
        return -1
    else:
        return 1
def gen_layout_metrics(data):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, 'eva.txt') 

    file1 = open(filepath, 'r')
    data = []
    for line in file1.readlines():
        line = line.strip()
        if line == "":
            continue
        if line[0] == "c":
            continue
        data.append(line)
    from functools import cmp_to_key

    data = sorted(data, key=cmp_to_key(compare))
    print(*data, sep = "\n")
    with open(filepath, 'w') as f:
        for line in data:
            f.write(line + "\n")
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
    
    l = LayoutRater()
    l.variational_flag=False
    train_data, test_data = l.loadData('train', "test_data_w_metrics_0")

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # tmp = []
    # for data in test_data:
    #     M_l = layout_metrics.edge_length_variation(data)
    #     print(data, M_l)
    # preprocess.save_data(tmp, 'test_data_w_metrics_698')
    #l.generate_model(train_loader, test_loader)
    if l.variational_flag:
        model = l.load_model('VGAE')
        print("VGAE")
    else:
        model = l.load_model('GAE')
        print("GAE")

    l.test(model=model,dataLoader=test_loader)







