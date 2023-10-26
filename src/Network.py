import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

class NetAll(nn.Module):
    def __init__(self):
        super(NetAll, self).__init__()

        self.num_layers = 1

        self.sep1 = nn.Linear(2, 64)
        self.sep2 = nn.Linear(13, 64)
        self.sep3 = nn.Linear(40, 128)
        self.sep4 = nn.Linear(365, 128)

        self.sep5 = nn.Linear(64, 64)
        self.sep6 = nn.Linear(64, 64)
        self.sep7 = nn.Linear(128, 128)
        self.sep8 = nn.Linear(128, 128)

        self.conv1 = GATConv(384-128, 128)
        self.conv2 = GATConv(128, 128)
        self.conv3 = GATConv(128, 64)
        
        self.lstm = nn.LSTM(20, 64, num_layers = self.num_layers)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 56)

        self.act1 = F.sigmoid()
        self.act2 = F.relu()
    
    def forward(self, data):
        x_all, edge_index = data.x, data.edge_index

        x_road = x_all[:, 1:3]
        x_location = x_all[:, 3:16]
        x_se = x_all[:, 16:56]
        x_svi = x_all[:, 56: 421]
        t = x_all[:, 421:].reshape((1, x_all.shape[0], 20))

        x_road = self.sep1(x_road)
        x_road = self.act1(x_road)
        x_road = self.sep5(x_road)
        x_road = self.act2(x_road)

        x_location = self.sep2(x_location)
        x_location = self.act1(x_location)
        x_location = self.sep6(x_location)
        x_location = self.act2(x_location)

        x_se = self.sep3(x_se)
        x_se = self.act1(x_se)
        x_se = self.sep7(x_se)
        x_se = self.act2(x_se)

        x_svi = self.sep4(x_svi)
        x_svi = self.act1(x_svi)
        x_svi = self.sep8(x_svi)
        x_svi = self.act2(x_svi)

        x = torch.cat((x_road, x_location, x_se, x_svi), 1)

        x = self.conv1(x, edge_index)
        x = self.act2(x)

        x = self.conv2(x, edge_index)
        x = self.act2(x)

        x = self.conv3(x, edge_index)
        x = self.act2(x)

        h0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x_all.shape[0], 64).requires_grad_().to(device)

        _, (hn, _) = self.lstm(t, (h0, c0))

        t = hn[0]
        x = torch.cat((x, t), 1)

        x = self.linear1(x)
        x = self.act2(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        return x