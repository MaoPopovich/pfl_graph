import torch
from torch_geometric.utils import to_networkx, from_networkx, add_self_loops, remove_self_loops, to_undirected
from torch_geometric.loader import ClusterData, ClusterLoader
import networkx as nx
import community as community_louvain
import random
import numpy as np

def louvain_splitter(data, client_num, delta=20, offset=0):
    """
    subgraph partition based on louvain algorithm
    """
    data.index_orig = torch.arange(data.num_nodes)
    G = to_networkx(
        data,
        node_attrs=['x', 'y'],
        to_undirected=True)
    nx.set_node_attributes(G,
                            dict([(nid, nid)
                                    for nid in range(nx.number_of_nodes(G))]),
                            name="index_orig")
    partition = community_louvain.best_partition(G)

    cluster2node = {}
    for node in partition:
        cluster = partition[node]
        if cluster not in cluster2node:
            cluster2node[cluster] = [node]
        else:
            cluster2node[cluster].append(node)   # cluster nodes based on louvain community detection: clusternode[cluster] = node_list

    max_len = len(G) // client_num - delta
    max_len_client = len(G) // client_num
    print("before:", len(cluster2node.keys()))
    tmp_cluster2node = {}
    for cluster in cluster2node:
        while len(cluster2node[cluster]) > max_len:
            tmp_cluster = cluster2node[cluster][:max_len]
            tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                1] = tmp_cluster
            cluster2node[cluster] = cluster2node[cluster][max_len:]
    cluster2node.update(tmp_cluster2node)

    orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
    orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

    print("after:",len(orderedc2n))
    client_node_idx = {idx: [] for idx in range(client_num)}
    idx = 0
    
    for (cluster, node_list) in orderedc2n:
        while len(node_list) + len(
                client_node_idx[idx]) > max_len_client + delta + offset:
            idx = (idx + 1) % client_num
        client_node_idx[idx] += node_list
        idx = (idx + 1) % client_num

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        graphs.append(from_networkx(nx.subgraph(G, nodes)))

    return graphs

def metis_splitter(data, client_num):
    """
    subgraph partition based on metis algorithm(PyG built-in)
    """
    graphs = []
    clusters = ClusterData(data, num_parts=client_num)
    loader = ClusterLoader(clusters, batch_size=1)

    for graph in loader:
        graphs.append(graph)
    
    return graphs

def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):  
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]

    return idx_batch

def hetero_splitter(data, client_num, alpha=0.5, classes=7):
    """
    subgraph partition based on dirichlet distribution
    """
    data.index_orig = torch.arange(data.num_nodes)
    G = to_networkx(data,
                    node_attrs=['x','y'],
                    to_undirected=True)
    nx.set_node_attributes(G,
                           dict([(nid,nid)
                                  for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    client_node_idx = {idx: [] for idx in range(client_num)}

    idx_batch = [[] for _ in range(client_num)]
    N = len(data.y)
    K = classes

    # create node index following dirichlet distribution for each classes
    for k in range(K):
        idx_k = np.where(data.y.numpy() == k)[0]
        idx_batch = partition_class_samples_with_dirichlet_distribution(
            N, alpha, client_num, idx_batch, idx_k
        )
    
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        client_node_idx[i] = idx_batch[i]
    
    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        graphs.append(from_networkx(sub_g))
    
    return graphs


def train_test_split(graphs, train_ratio=0.01):
    """
    split train/test set with a fixed ratio for each subgraph
    """

    for graph in graphs:
        # To undirected and add self_loop
        graph.edge_index = add_self_loops(to_undirected(remove_self_loops(graph.edge_index)[0]),num_nodes=graph.x.shape[0])[0]
        node_num = graph.x.shape[0]
        node_idx = np.arange(node_num)
        np.random.shuffle(node_idx)

        train_mask = torch.zeros_like(graph.y, dtype=torch.bool)
        test_mask = torch.zeros_like(graph.y, dtype=torch.bool)

        train_idx, test_idx = node_idx[:int(train_ratio*node_num)], node_idx[int(train_ratio*node_num):]
        
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        # set train/test mask for each subgraph
        graph.train_mask = train_mask
        graph.test_mask = test_mask

    return graphs


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True