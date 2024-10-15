import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import utility_functions
import os
import sys


proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

label_path=proj_dir+'\\data\\raw\\label.csv'
adj_path=proj_dir+'\\data\\adj_matrices\\adj_matrix_4neig_None_correlation.csv'
raw_path=proj_dir+'\\data'
data=pd.read_csv(proj_dir+'\\data\\raw\\input_table.csv')
data=data.drop(labels=['pixel_indices','lat','lon'],axis=1)
scaler = utility_functions.get_scaler(data)

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['input_table.csv']

    @property
    def processed_file_names(self):
        return ['input_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        idx = 0

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.

            node_features = pd.read_csv(raw_path)
            node_features=node_features.drop(labels=['pixel_indices','lat','lon'],axis=1)
            features=node_features.to_numpy()
            node_features= scaler.transform(features)
            node_feats = torch.tensor(node_features, dtype=torch.float)
            label_feats = self._get_label(label_path)
            edge_index = self._get_adjacency_info(adj_path)
            data = Data(x=node_feats, edge_index=edge_index, y=label_feats)



            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def _get_adjacency_info(self, adj_path):
        data = pd.read_csv(adj_path)
        adj_matrix = data.to_numpy()
        edge_indices = torch.tensor(adj_matrix)

        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_label(self, label_path):
        data = pd.read_csv(label_path)

        label = data.to_numpy()#.transpose()

        label_feats = torch.tensor(label, dtype=torch.long)


        return label_feats

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

dataset = MyOwnDataset(root = raw_path)
torch.save(dataset , proj_dir+'\\data\\graphs\\train_data_None_correlation_4neig.pth')