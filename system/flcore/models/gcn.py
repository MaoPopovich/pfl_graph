import torch
import torch.nn.functional as F 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x

class GCN_PER(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64):
        super(GCN_PER, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.fc = GCNConv(hidden, out_channels)

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.fc(x)

        return x

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, data):
        out = self.base(data)
        out = self.head(out, data.edge_index)

        return out