import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, in_channels: int,
                 num_hidden: int):
        super(Encoder, self).__init__()
        # define 2-layer GCN
        self.conv1 = GCNConv(in_channels, 2*num_hidden)
        self.conv2 = GCNConv(2*num_hidden, num_hidden)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.rrelu(x)
        x = self.conv2(x, edge_index)
        return x 


class GRACE(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int, 
                 num_hidden: int, 
                 num_proj_hidden: int,  
                 tau: float):
        super(GRACE, self).__init__()
        # define base encoder
        self.base = Encoder(in_channels, num_hidden)

        # define projection layers
        self.proj = nn.Sequential(nn.Linear(num_hidden, num_proj_hidden), nn.ELU(), nn.Linear(num_proj_hidden, num_hidden))

        # define classifier
        self.cls = nn.Linear(num_hidden, out_channels)

        self.tau = tau

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        
        # graph convolution operation
        z = self.base(x, edge_index)
        return z 

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    # We only define contrastive loss inside GRACE 
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    # We only implement classifier inside GRACE, the calculation of CrossEntropy is implemented in Client.train()
    def generate_logits(self, z: torch.Tensor):
        return self.cls(z)
