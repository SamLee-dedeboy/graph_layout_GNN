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

# generate train and test dataset and save to file.
# train_graph_list: specifies raw data files of train graphs, e.g. 0.edges
# test_graph_list: specifies raw data files of test graphs
# test_layout_func_list: specifies functions for layout algorithms, e.g. nx.spiral_layout
def generateDataset(train_graph_list, test_graph_list, test_layout_func_list):
    train_data=[]
    test_data=[]
    #
    # assemble graph & layout data into one dataset
    #
    # train dataset
    for g in train_graph_list:
        graphData = readRawData("../dataset/facebook/" + str(g) + ".edges")
        graph_data_list = sub_sample_large_graph(graphData, g)
        for index, sub_graph_data in enumerate(graph_data_list):
            sub_graph_data = add_neg_edges(sub_graph_data)
            for j in range(len(test_layout_func_list)):
                pos = generatePos(sub_graph_data, test_layout_func_list[j])
                data = add_position(sub_graph_data, pos)
                data.num_nodes=data.x.size()[0]
                data.graph_tag = [g, index]
                data.layout_tag = j
                train_data.append(data)
                print(data)
    # test dataset
    for g in test_graph_list:
        graphData = readRawData("../dataset/facebook/" + str(g) + ".edges")
        graphData = add_neg_edges(graphData)
        for j in range(len(test_layout_func_list)):
            pos = generatePos(graphData, test_layout_func_list[j])
            data = add_position(graphData, pos)
            data.num_nodes=data.x.size()[0]
            data.graph_tag = g
            data.layout_tag = j
            #data = add_readability_metrics(data)
            print(data)
            test_data.append(data)

    # save to file 'train' and 'test'
    save_data(train_data, 'train')
    save_data(test_data, 'test')

# saves data to file
def save_data(data, filename):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, filename) 
    torch.save(data, filepath)

# add readability metrics to existing data object.
# calculating E_c could take very long.
def add_readability_metrics(data):
    M_l = layout_metrics.edge_length_variation(data.cpu())
    M_a = layout_metrics.minimum_angle(data.cpu())
    print("calculating E_c...")
    E_c = layout_metrics.edge_crossings(data.cpu())
    data.M_l = M_l
    data.M_a = M_a
    data.E_c = E_c
    return data

# sub-sample(partition) large graphs(> 500 nodes) into smaller graphs(~130 nodes)
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
# add position matrix to data object.
# pos is assumed to be an array of size [num_nodes, 2]
def add_position(graphData, pos):
    data = graphData.clone()
    
    # uses node-pair distance for position matrix. used in preliminary experiments
    #x = torch.tensor(generateNodeDist(pos), dtype=torch.float)

    # uses node coordinates in 2d space as node position matrix
    x = torch.tensor(pos, dtype=torch.float)
    data.x=x
    return data

# read from raw data. example filename: 0.edges
def readRawData(filename):
    pos_edges = []
    base_path = Path(__file__).parent
    file_path = (base_path / filename).resolve()      
    with open(file_path) as f:
        lines = f.readlines()
        # node dict is used to map node id in raw data to [0, num_nodes-1].
        # otherwise model will not function properly
        node_dict={}
        # construct list of edges
        for line in lines:
            new_edge = line.split(" ")
            new_edge = list(map(lambda s: int(s.strip()), new_edge))
            if(not new_edge[0] in node_dict.keys()):
                node_dict[new_edge[0]]=len(node_dict)
            if(not new_edge[1] in node_dict.keys()):
                node_dict[new_edge[1]]=len(node_dict)
            pos_edges.append([node_dict[new_edge[0]], node_dict[new_edge[1]]]) 
        edge_index = torch.tensor(pos_edges, dtype=torch.int64).t().contiguous()

        # x is one-hot encoding of nodes for now. will be replaced with node position matrix later
        x=[[i] for i in range(len(node_dict))]
        x=torch.tensor(x, dtype=torch.float32)
        data = Data(edge_index=edge_index,x=x)

        assert data.edge_index.max() < data.num_nodes
        return data

# add negative edges(non-existing edges) to perform link prediction
def add_neg_edges(data):
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
    
    edge_index = torch.cat((pos_edge_index, neg_edge_index), 1).t().contiguous()
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

    data.edge_index = edge_index
    data.pos_edge_index = pos_edge_index
    data.neg_edge_index = neg_edge_index
    
    assert data.edge_index.max() < data.num_nodes
    return data

# generate node positions from the targeted layout algorithm
# layout_func is expected to return pos, 
# which is a dictionary of normalized 2d node coordinates, key is node id
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


# generate node-pair distances from node positions.
# deprecated, used in preliminary experiments
def generateNodeDist(pos):
    node_num = len(pos)
    pos_in_pair = [[np.linalg.norm(pos[i]-pos[j]) for j in range(node_num)] for i in range(node_num)]
    return pos_in_pair  

# show the layout image of the data.
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

# converts edge_index, an array of [2, edge_num] used in geometric.data to a networkx graph
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

# codes to generate train & test dataset. 
# uncomment and run if want to add or regenerate data.
# train_graph_list = [107, 348, 686, 1684, 1912, 3437]

# test_graph_list = [0, 414, 3980, 698]

# test_layout_func_list = [MDS.mds_layout, nx.spiral_layout, nx.spring_layout, nx.circular_layout]

# generateDataset(train_graph_list, test_graph_list, test_layout_func_list)