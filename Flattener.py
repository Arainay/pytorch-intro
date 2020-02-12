import torch.nn as nn


class Flattener(nn.Module):
    def forward(self, x):
        batch_size, _ = x.shape

        return x.view(batch_size, -1)
