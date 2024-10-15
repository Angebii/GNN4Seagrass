from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class GCN_1ly(torch.nn.Module):
    def __init__(self, in_dim,hidden_channels1):
        super(GCN_1ly, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_channels1)
        self.linear1 = torch.nn.Linear(hidden_channels1,1)



    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x1= self.linear1(x)
        x1=F.sigmoid(x1)
        return x1