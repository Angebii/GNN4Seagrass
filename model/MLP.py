import torch

class MLP(torch.nn.Module):
    def __init__(self, in_dim,hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_channels,1)


    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = torch.functional.F.dropout(x, p=0.5, training=self.training)
        x1= self.linear1(x)

        x1=torch.functional.F.sigmoid(x1)

        return x1