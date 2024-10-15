from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class GCN_3ly(torch.nn.Module):
    def __init__(self, in_dim,hidden_channels1,hidden_channels2,hidden_channels3):
        super(GCN_3ly, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, hidden_channels3)
        self.linear1 = torch.nn.Linear(hidden_channels3,1)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x1= self.linear1(x)

        x1=F.sigmoid(x1)

        return x1