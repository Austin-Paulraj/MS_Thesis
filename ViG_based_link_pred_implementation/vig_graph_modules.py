import torch
from torch import nn
from torch_geometric import nn as gnn
from timm.models.layers import DropPath
import sys

sys.path.append('/home/paulraae/MS_Thesis/ViG_based_link_pred_implementation')
from vig_based_functions import act_layer, get_neighbor_coo

class Stem(nn.Module):
    def __init__(self, in_dim=3, out_dim=48, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_features, act='relu', drop_path=0.0):
        super().__init__()
        out_features = in_features
        hidden_features = in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class Grapher(nn.Module):
    def __init__(self, in_channels, img_dim, at_heads = 4, act='relu', drop_path=0.0, device = "cuda:7"):
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = gnn.GATv2Conv(in_channels*img_dim*img_dim, in_channels*img_dim*img_dim, heads = at_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * at_heads, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.device = device

    def forward(self, x, NUM_NEIGHBORS, EDGE_METHOD):
        _tmp = x
        B,C,H,W = x.shape
        x = self.fc1(x)
        edge_index = get_neighbor_coo(x, NUM_NEIGHBORS, method = EDGE_METHOD)
        edge_index = torch.Tensor(edge_index).to(torch.int32).to(self.device)
        x = x.reshape(B,-1)
        x = self.graph_conv(x, edge_index)
        del edge_index
        x = x.reshape(B,-1,H,W)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x

class GrapherSetEdges(nn.Module):
    def __init__(self, in_channels, img_dim, at_heads = 4, drop_path=0.0, device = "cuda:7"):
        super(GrapherSetEdges, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = gnn.GATv2Conv(in_channels*img_dim*img_dim, in_channels*img_dim*img_dim, heads = at_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * at_heads, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.device = device

    def forward(self, x, edge_index):
        _tmp = x
        B,C,H,W = x.shape
        x = self.fc1(x)
        x = x.reshape(B,-1)
        x = self.graph_conv(x, edge_index)
        x = x.reshape(B,-1,H,W)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
  
class GraphEncoderBlock(nn.Module):
    def __init__(self, in_channels = 3, img_dim = 56, at_heads = 4, act='relu', drop_path=0.0, device = "cuda:7"):
        super(GraphEncoderBlock, self).__init__()
        self.grapher = Grapher(in_channels, img_dim, at_heads, drop_path, device)
        self.ffn = FFN(in_channels, act, drop_path)


    def forward(self, x, NUM_NEIGHBORS, EDGE_METHOD):
        x = self.grapher(x, NUM_NEIGHBORS, EDGE_METHOD)
        return self.ffn(x)
    
class GraphEncoderBlockSE(nn.Module):
    def __init__(self, in_channels = 3, img_dim = 56, at_heads = 4, act='relu', drop_path=0.0, device = "cuda:7"):
        super(GraphEncoderBlockSE, self).__init__()
        self.grapher = GrapherSetEdges(in_channels, img_dim, at_heads, drop_path, device)
        self.ffn = FFN(in_channels, act, drop_path)


    def forward(self, x, edge_index):
        x = self.grapher(x, edge_index)
        return self.ffn(x)