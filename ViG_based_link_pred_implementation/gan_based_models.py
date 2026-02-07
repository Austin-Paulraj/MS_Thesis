import torch
from torch import nn
import sys

sys.path.append('/home/paulraae/MS_Thesis/ViG_based_link_pred_implementation')
from vig_based_functions import act_layer

class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, act='relu'):
      super().__init__()
      self.convs = nn.Sequential(
          nn.Conv2d(in_dim, out_dim//4, 3, stride=2, padding=1),
          nn.BatchNorm2d(out_dim//4),
          act_layer(act),
          nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
          nn.BatchNorm2d(out_dim//2),
          act_layer(act),
          nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
          nn.BatchNorm2d(out_dim),
          act_layer(act),
          nn.Flatten(),
          nn.Linear(out_dim*7*7, 4096),
          act_layer(act),
          nn.Linear(4096, 1),
      )

    def forward(self, x):
      x = self.convs(x)
      x = x.reshape(x.shape[0])
      return x