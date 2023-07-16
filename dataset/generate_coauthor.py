import torch
import numpy as np
import copy
import argparse
import os
import pickle 
import random

from torch_geometric.datasets import Coauthor
from torch_geometric.data import Data 

from data_utils import louvain_splitter, metis_splitter, hetero_splitter, train_test_split, setup_seed

INF = np.iinfo(np.int64).max


def load_nodelevel_dataset(args):
    """
    returns: data_dict
    rtype: Dict: dict{'client_id': Data()}
    """
    # load dataset
    name, client_num, algo, alpha = args.name, args.client_num, args.algorithm, args.alpha
    if name in ["CS", "Physics"]:
        dataset = Coauthor(root='dataset',
                            name=name)
        # no default mask
        # del dataset[0].train_mask
        # del dataset[0].val_mask
        # del dataset[0].test_mask

        # global_dataset = copy.deepcopy(dataset)
        global_dataset = None
    # split a global graph into subgraphs
    if algo == 'louvain':
        dataset = louvain_splitter(data=dataset[0], client_num=client_num, offset=1000)
    elif algo == 'metis':
        dataset = metis_splitter(data=dataset[0], client_num=client_num)
    elif algo == 'hetero':
        classes = int(dataset[0].y.max())+1
        dataset = hetero_splitter(data=dataset[0], client_num=client_num, alpha=alpha, classes=classes)


    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), client_num) if client_num > 0 else len(dataset)

    # train/val/test split for each subgraph
    train_ratio = args.train_ratio
    train_test_split(dataset, train_ratio=train_ratio)

    # get local dataset
    data_dict = dict()
    for client_idx in range(len(dataset)):
        local_data = dataset[client_idx]
        data_dict[client_idx] = local_data

    # Keep ML split consistent with local graphs
    if global_dataset is not None:
        global_graph = global_dataset[0]
        train_mask = torch.zeros_like(global_graph.y, dtype=torch.bool)
        test_mask = torch.zeros_like(global_graph.y, dtype=torch.bool)

        for client_sampler in data_dict.values():
            if isinstance(client_sampler, Data):
                client_subgraph = client_sampler
            else:
                client_subgraph = client_sampler['data']
            train_mask[client_subgraph.index_orig[
                client_subgraph.train_mask]] = True
            test_mask[client_subgraph.index_orig[
                client_subgraph.test_mask]] = True
        global_graph.train_mask = train_mask
        global_graph.test_mask = test_mask

        data_dict[len(dataset)] = global_graph
    
    # save subgraphs into disk
    save_path = os.path.join('dataset', name, 'subgraph', algo+str(train_ratio))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for idx, graph in data_dict.items():
        with open(save_path+'/'+str(idx)+'.pkl', 'wb') as f:
            pickle.dump(graph, f)

    print("Finish generating subgraphs!")
    return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--name", type=str, default="CS",
                        help="specify dataset name")
    parser.add_argument('-c', "--client_num", type=int, default=10,
                        help="specify client number")
    parser.add_argument('-algo', "--algorithm", type=str, default="louvain",
                        help="specify subgraph partition algorithm")
    parser.add_argument('-train',"--train_ratio", type=float, default=0.4,
                        help="specify the ratio of train/test set")
    parser.add_argument('-alpha', type=float, default=0.5,
                        help="specify the factor of dirichlet distribution")
    args = parser.parse_args()

    setup_seed(seed=0)
    subgraphs = load_nodelevel_dataset(args)
    print(len(subgraphs[len(subgraphs)-1].y))