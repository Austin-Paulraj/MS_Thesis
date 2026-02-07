import torch
from torch import nn
import sys

sys.path.append('/home/paulraae/MS_Thesis/ViG_based_link_pred_implementation')
from vig_based_functions import act_layer, get_multi_shot_set
from vig_graph_modules import Stem, GraphEncoderBlock

class FullModel(nn.Module):
    def __init__(self, num_slices = 16, num_shots = 3, num_of_heads = 2, num_blocks_per_head = 2, in_channels = 3, out_channels = 48, img_dim = 56, at_heads = 4, act='relu', drop_path=0.0, device="cuda:7", deletion_threshold = 0.5):
      super(FullModel, self).__init__()
      self.encoder_blocks = GraphEncoderBlock(out_channels, img_dim, at_heads, act, drop_path, device).to(device)
      self.encoder_block2 = GraphEncoderBlock(out_channels, img_dim, at_heads, act, drop_path, device).to(device)
      #self.gen_encode = Encode(in_channels, out_channels, act)
      self.gen_decode = Decode(in_channels, (out_channels*(num_shots+1)*num_slices), act)
      #self.gen_decode = Big_Decoder((out_channels*num_shots*num_slices), in_channels)
      self.stem = Stem(in_channels, out_channels)
      self.deletion_threshold = deletion_threshold
      self.num_shots = num_shots
      self.img_dim = img_dim
      self.device = device

    def forward(self, query_image, x, T, NUM_NEIGHBORS, EDGE_METHOD):
      x = self.stem(x)
      indeces = torch.Tensor([i for i in range(x.shape[0]-16)])
      x = self.encoder_blocks(x, NUM_NEIGHBORS, EDGE_METHOD)
      x = self.encoder_block2(x, NUM_NEIGHBORS, EDGE_METHOD)

      n_shot_indices = []
      full_set = get_multi_shot_set(x, T, self.num_shots)
      full_set = full_set.reshape(-1, self.img_dim, self.img_dim)

      return self.gen_decode(full_set.unsqueeze(0))

class Decode(nn.Module):
    def __init__(self, in_dim=3, out_dim=240, act='relu'):
        super().__init__()
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim//2, 2, stride=2),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.ConvTranspose2d(out_dim//2, out_dim//4, 2, stride=2),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.ConvTranspose2d(out_dim//4, out_dim//8, 2, stride=2),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.ConvTranspose2d(out_dim//8, out_dim//16, 2, stride=2),
            nn.BatchNorm2d(out_dim//16),
            act_layer(act),
            nn.ConvTranspose2d(out_dim//16, out_dim//16, 2, stride=2),
            nn.BatchNorm2d(out_dim//16),
            act_layer(act),
            nn.ConvTranspose2d(out_dim//16, in_dim, 1, stride=1),
            nn.Conv2d(in_dim, in_dim, 1, 1),
        )

    def forward(self, x):
        x = self.deconvs(x)
        x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
        return x