from sklearn.neighbors import KNeighborsTransformer
import copy
import cv2
import torch
from torch import nn
import numpy as np

def split_grid(im, n=4):
  B,C,H,W = im.shape
  ln = int(H/n)
  image = cv2.resize(image,(ln*n, ln*n))
  H = ln
  W = ln
  tiles = [im[x:x+H,y:y+W] for x in range(0,im.shape[0],H) for y in range(0,im.shape[1],W)]

  return torch.Tensor(tiles)

def tile_split(image, n=2):
    ln = int(224/n)
    image = cv2.resize(image,(ln*n, ln*n))
    
    H = ln
    W = ln
    im2 = copy.deepcopy(image)
    tiles = [im2[x:x+H,y:y+W] for x in range(0,im2.shape[0],H) for y in range(0,im2.shape[1],W)]
    
    return np.array(tiles)


def coo_one_2_two(edge_map):
  edge_map_aux = [[],[]]
  for x in edge_map:
      edge_map_aux[0].append(x[0])
      edge_map_aux[1].append(x[1])

  edge_map_aux = np.array(edge_map_aux)
  return(torch.Tensor(edge_map_aux))



def adjacency_to_coo(adj_matrix):
  rows, cols = np.where(adj_matrix != 0)
  non_diagonal_mask = rows != cols

  filtered_rows = rows[non_diagonal_mask]
  filtered_cols = cols[non_diagonal_mask]

  coo_list = np.stack([filtered_rows, filtered_cols], axis=1)
  return coo_one_2_two(coo_list)



def get_neighbor_coo(data, neighbors = 12, method = "KNN"):
  B,C,H,W = data.shape
  x = data.detach().cpu()
  if method == "KNN":
    transformer = KNeighborsTransformer(n_neighbors=neighbors, mode='connectivity')
    graph = transformer.fit_transform(x.reshape(B,-1)).toarray()
  del x
  return adjacency_to_coo(graph)


def get_past_threshold(data, n_slices = 16, threshold = 0.1):
  distances = []
  avg = torch.mean(data[:n_slices])
  for img in data[n_slices:]:
    distances.append(torch.dist(avg, img))

  distances = torch.Tensor(distances)
  distances = distances/distances.max()
  indices = torch.where(distances < threshold)[0]

  return torch.cat([data[:n_slices], data[n_slices:][indices]], dim = 0), indices

def get_multi_shot_set(data, n_slices = 16, n_shots = 5):
  distances = []
  shot_set = []
  shot_indeces = []
  avg = torch.mean(data[:n_slices])
  for img in data[n_slices:]:
    distances.append(torch.dist(avg, img))

  distances = torch.Tensor(distances)
  values, indices = torch.topk(distances, n_slices*n_shots, largest=False, sorted=True)
  full_set = data[n_slices:][indices]
  
  full_set = torch.cat([data[:n_slices], full_set], dim=0)

  for i in range(n_shots):
    shot_set.append(full_set[i*n_slices: (i+1)*n_slices].to("cpu"))
    shot_indeces.append(indices[i*n_slices: (i+1)*n_slices].to("cpu"))



  return full_set#, torch.Tensor(np.array(shot_set)), indices.to(torch.int), torch.Tensor(np.array(shot_indeces)).to(torch.int)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer