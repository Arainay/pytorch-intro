import torch.nn as nn


class Flattener(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]

        return x.view(batch_size, -1)
