import torch
from torch_geometric.data import Data
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.utils.convert as cv
from torch_geometric.utils import negative_sampling
import numpy as np
import random
import os
import MDS
import layout_metrics
#from geo_gcn.src.architectures import SGCN
sub_sample_rate = 0.1
def generateDataset(graph_list, test_graph_list, test_layout_func_list):
    use_spatial=False
    #num_of_graph = len(graph_list)
    train_data=[]
    test_data=[]
    # assemble graph & layout data into one dataset
    #for i in range(num_of_graph):
    # train dataset
    for g in graph_list:
        graphData = readRawData("../dataset/facebook/" + str(g) + ".edges")
        graph_data_list = sub_sample_large_graph(graphData, g)
        for index, sub_graph_data in enumerate(graph_data_list):
            sub_graph_data = add_neg_edges(sub_graph_data)
            for j in range(len(test_layout_func_list)):
                pos = generatePos(sub_graph_data, test_layout_func_list[j])
                data = add_position(sub_graph_data, pos, use_spatial)
                data.num_nodes=data.x.size()[0]
                data.graph_tag = [g, index]
                data.layout_tag = j
                train_data.append(data)
                print(data)
           
    for g in test_graph_list:
        graphData = readRawData("../dataset/facebook/" + str(g) + ".edges")
        graphData = add_neg_edges(graphData)
        for j in range(len(test_layout_func_list)):
            pos = generatePos(graphData, test_layout_func_list[j])
            data = add_position(graphData, pos, use_spatial)
            data.num_nodes=data.x.size()[0]
            data.graph_tag = g
            data.layout_tag = j
            #data = add_readability_metrics(data)
            print(data)
            test_data.append(data)
            #print(data)
    save_data(train_data, 'train')
    save_data(test_data, 'test')
    # dirname = os.path.dirname(__file__)
    # train_filepath = os.path.join(dirname, 'train') 
    # test_filepath = os.path.join(dirname, 'test') 
    # torch.save(train_data, train_filepath)
    # torch.save(test_data, test_filepath)

def save_data(data, filename):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, filename) 
    torch.save(data, filepath)
def add_readability_metrics(data):
    M_l = layout_metrics.edge_length_variation(data.cpu())
    M_a = layout_metrics.minimum_angle(data.cpu())
    print("calculating E_c...")
    E_c = layout_metrics.edge_crossings(data.cpu())
    data.M_l = M_l
    data.M_a = M_a
    data.E_c = E_c
    return data
def sub_sample_large_graph(graphData, g):
    sub_graph_list = []
    if(graphData.num_nodes > 500):
        num_of_nodes = graphData.num_nodes
        partition = num_of_nodes/130
        for i in range(round(partition)):
            subset_num_of_nodes = 130 + random.randrange(-20,20)
            subset = torch.tensor(np.random.choice(range(num_of_nodes), size=subset_num_of_nodes), dtype=torch.int64)
            sub_graph = graphData.subgraph(subset)
            sub_graph_list.append(sub_graph)
        return sub_graph_list
    else:
        return [graphData]
def add_position(graphData, pos, use_SGCN):
    data = graphData.clone()
    if use_SGCN == True:
        data.pos=torch.tensor(np.array(pos), dtype=torch.float)
        #data = Data(x=data.x, pos=torch.tensor(np.array(pos), dtype=torch.float), edge_index=data.edge_index, pos_edge_index = data.pos_edge_index)
    else:
        #x = torch.tensor(generateNodeDist(pos), dtype=torch.float)
        x = torch.tensor(pos, dtype=torch.float)
        data.x=x
        #data = Data(x=x, edge_index=data.edge_index, pos_edge_index = data.pos_edge_index)
    return data
def readPosition(filename, node_dict):
    import json
    base_path = Path(__file__).parent
    file_path = (base_path / filename).resolve()      
    with open(file_path) as f:
        posDict = json.load(f)
        width = posDict["CanvasSize"][0]
        height = posDict["CanvasSize"][1]

        posDict.pop("CanvasSize")
        pos = [None]*len(node_dict)
        for k, v in posDict.items():
            k=int(k)
            pos[node_dict[k]] = [v[0]/width, v[1]/height]

    return pos        
        
def readRawData(filename):
    pos_edges = []
    base_path = Path(__file__).parent
    file_path = (base_path / filename).resolve()      
    with open(file_path) as f:
        lines = f.readlines()
        # construct list of edges
        node_dict={}
        for line in lines:
            new_edge = line.split(" ")
            new_edge = list(map(lambda s: int(s.strip()), new_edge))
            if(not new_edge[0] in node_dict.keys()):
                node_dict[new_edge[0]]=len(node_dict)
            if(not new_edge[1] in node_dict.keys()):
                node_dict[new_edge[1]]=len(node_dict)
            pos_edges.append([node_dict[new_edge[0]], node_dict[new_edge[1]]]) 
        edge_index = torch.tensor(pos_edges, dtype=torch.int64).t().contiguous()

        x=[[i] for i in range(len(node_dict))]
        x=torch.tensor(x, dtype=torch.float32)
        data = Data(edge_index=edge_index,x=x)
        # G = cv.to_networkx(data, to_undirected=True)
        # x = nx.adjacency_matrix(G).todense()
        # data = Data(edge_index=edge_index.t().contiguous(), x=x)
        #print(data)
        assert data.edge_index.max() < data.num_nodes
        return data

def add_neg_edges(data):
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
    
    #edge_index = random.sample(pos_edges, round(len(pos_edges)*sub_sample_rate))
    edge_index = torch.cat((pos_edge_index, neg_edge_index), 1).t().contiguous()
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

    data.edge_index = edge_index
    data.pos_edge_index = pos_edge_index
    data.neg_edge_index = neg_edge_index
    # G = cv.to_networkx(data, to_undirected=True)
    # x = nx.adjacency_matrix(G).todense()
    # data = Data(edge_index=edge_index.t().contiguous(), x=x)
    #print(data)
    assert data.edge_index.max() < data.num_nodes
    return data

def generatePos(graphData, layout_func, seed=None):
    plt.figure(1, figsize=(300, 180), dpi=60)
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])
    tmp_data = Data(x=graphData.x, edge_index=graphData.pos_edge_index)
    G = cv.to_networkx(tmp_data, to_undirected=True)
    if seed == None:
        pos = layout_func(G, center=(0.5,0.5), scale=0.5)
    else:
        pos = nx.random_layout(G, seed=seed)
    # pos is a dict, need to convert to list
    pos_as_list = [pos[key] for key in sorted(pos.keys())]
    assert np.min(pos_as_list)>=0 and np.max(pos_as_list)<=1
    return pos_as_list

def drawGraph(graphData, layout_func):
    options = {
    'node_color': 'blue',
    'node_size': 20,
    'width': 1,
    }
    plt.figure(1, figsize=(300, 180), dpi=60)
    G = cv.to_networkx(graphData, to_undirected=True)
    nx.draw(G, pos=layout_func(G), **options)
    plt.show()

def partition(data):
    length = len(data)
    pivot = round(length*0.8)
    train_data = data[0:pivot]
    test_data = data[pivot:]
    return train_data, test_data
def generateNodeDist(pos):
    node_num = len(pos)
    pos_in_pair = [[np.linalg.norm(pos[i]-pos[j]) for j in range(node_num)] for i in range(node_num)]
    return pos_in_pair  
    
def showLayout(dataLoader):
    options = {
    'node_color': 'blue',
    'node_size': 20,
    'width': 1,
    }
    plt.figure(1, figsize=(300, 180), dpi=60)
    for data in dataLoader:
        tmp_data = Data(x=data.x, edge_index=data.pos_edge_index)
        G = cv.to_networkx(tmp_data, to_undirected=True)
        layout_index = data.layout_tag.item()
        layout_func = test_layout_func_list[layout_index]
        subax = plt.subplot(2, 2, layout_index+1)
        nx.draw(G, pos=layout_func(G), **options)
    plt.show()
    # pos is a dict, need to convert to list
def edge_index_to_graph(n_nodes, edge_index, p_edge_index):
    #print(p_edge_index.shape, edge_index.shape)
    recon_edge_index = []
    for count, edge in enumerate(np.transpose(edge_index)):
        #print(edge)
        if p_edge_index[count] > 0.5:
            recon_edge_index.append(edge.copy())
    #print(recon_edge_index)
    x=[i for i in range(n_nodes)]  
    recon_edge_index=torch.tensor(np.array(recon_edge_index), dtype=torch.int64)
    data = Data(edge_index=recon_edge_index.t().contiguous(),x=x)
    data.num_nodes = n_nodes
    assert data.num_nodes == n_nodes
    G = cv.to_networkx(data, to_undirected=True)
    return G


train_graph_list = [107, 348, 686, 1684, 1912, 3437]
#test_graph_list = [3980, 698]
test_graph_list = [0, 414, 3980, 698]

test_layout_func_list = [MDS.mds_layout, nx.spiral_layout, nx.spring_layout, nx.circular_layout]

#generateDataset(train_graph_list, test_graph_list, test_layout_func_list)