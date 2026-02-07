import numpy as np
import os
from torch_geometric.data import Dataset, download_url, Data
import torch

from torch_geometric import nn as gnn
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from torch import nn

class OneGraphDS(Dataset): #For this ds I have ald done all the required pre prcessing 
    def __init__(self, root, num_neighbors = 12, data_dict = None):
        self.data_dict = data_dict
        super().__init__(root, None, None, None)
        
        
    @property
    def raw_file_names(self):
        return "not_implementated.pt"

    @property
    def processed_file_names(self):
        return "not_implemented.pt"
    
    def process(self):
            # Process pre made data dictionary

            centroids = torch.Tensor(self.data_dict["tile_centroids"].reshape((12,56,56,3)))
            tiles = torch.Tensor(self.data_dict["tiles"].reshape((3232,56,56,3)))
            tile_clusters = torch.Tensor(self.data_dict["tile_cluster"])
            centroid_y = torch.Tensor(torch.Tensor([i for i in range(self.data_dict["tile_centroids"].shape[0])]))
            tile_knn_edges = torch.Tensor(self.data_dict["tile_knn_edge_index"])
            
            x_features = torch.cat((tiles, centroids))
            y = torch.cat((tile_clusters, centroid_y)).to(torch.int)
            
            edge_map = []


            i=-1
            for cluster in y[:tiles.shape[0]]:
                i+=1
                edge_map.append([i, cluster+tiles.shape[0]])
            

            i+=1
            for centroid1 in y[i:]:
                for centroid2 in y[i+1:]:
                    edge_map.append([centroid1+tiles.shape[0], centroid2+tiles.shape[0]])
                i+=1
            
            edge_map = torch.cat((torch.Tensor(edge_map), tile_knn_edges), axis=0)
            
            edge_map = edge_map.to(int)
            
            edge_map_aux = [[],[]]
            for x in edge_map:
                edge_map_aux[0].append(x[0])
                edge_map_aux[1].append(x[1]) 
                    
            edge_map_aux = np.array(edge_map_aux)
            
            edge_attrs = []
            for _ in edge_map_aux[0]:
                edge_attrs.append(1)
                
            edge_attrs = torch.Tensor(edge_attrs)
            edge_map_aux = torch.Tensor(edge_map_aux).to(int)
            
            data = Data(x_features, edge_map_aux, edge_attrs, y)

            torch.save(data, os.path.join(self.processed_dir, f'data_0.pt'))

    def len(self):
        return self.data_dict["features"].shape[0]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data