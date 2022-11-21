'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossNet(nn.Module):
    def __init__(self, num_layers, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        self.num_layers = num_layers

        if self.num_layers >= 1:
            self.GAP1 = nn.AvgPool2d(feature_sizes[0])
            self.FC1 = nn.Linear(num_channels[0], interm_dim)
        if self.num_layers >= 2:
            self.GAP2 = nn.AvgPool2d(feature_sizes[1])
            self.FC2 = nn.Linear(num_channels[1], interm_dim)
        if self.num_layers >= 3:
            self.GAP3 = nn.AvgPool2d(feature_sizes[2])
            self.FC3 = nn.Linear(num_channels[2], interm_dim)
        if self.num_layers >= 4:
            self.GAP4 = nn.AvgPool2d(feature_sizes[3])
            self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(self.num_layers * interm_dim, 1)

    def forward(self, features):
        assert len(features) == self.num_layers

        if self.num_layers == 1:
            out1 = self.GAP1(features[0])
            out1 = out1.view(out1.size(0), -1)
            out1 = F.relu(self.FC1(out1))
            out = self.linear(out1)
        elif self.num_layers == 2:
            out1 = self.GAP1(features[0])
            out2 = self.GAP2(features[1])
            out1 = out1.view(out1.size(0), -1)
            out1 = F.relu(self.FC1(out1))
            out2 = out2.view(out2.size(0), -1)
            out2 = F.relu(self.FC2(out2))
            out = self.linear(torch.cat((out1, out2), 1))
        elif self.num_layers == 3:
            out1 = self.GAP1(features[0])
            out2 = self.GAP2(features[1])
            out3 = self.GAP3(features[2])
            out1 = out1.view(out1.size(0), -1)
            out1 = F.relu(self.FC1(out1))
            out2 = out2.view(out2.size(0), -1)
            out2 = F.relu(self.FC2(out2))
            out3 = out3.view(out3.size(0), -1)
            out3 = F.relu(self.FC3(out3))
            out = self.linear(torch.cat((out1, out2, out3), 1))
        elif self.num_layers == 4:
            out1 = self.GAP1(features[0])
            out2 = self.GAP2(features[1])
            out3 = self.GAP3(features[2])
            out4 = self.GAP4(features[3])
            out1 = out1.view(out1.size(0), -1)
            out1 = F.relu(self.FC1(out1))
            out2 = out2.view(out2.size(0), -1)
            out2 = F.relu(self.FC2(out2))
            out3 = out3.view(out3.size(0), -1)
            out3 = F.relu(self.FC3(out3))
            out4 = out4.view(out4.size(0), -1)
            out4 = F.relu(self.FC4(out4))
            out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out
