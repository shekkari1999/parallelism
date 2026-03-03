import torch
import torch.nn as nn

## some params
LAYERS = 16
HIDDEN_DIM = 128
SEED = 42
BATCH_SIZE = 32
STEPS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#--------------------------------------------------------------------------------------------------------------------------
class ShardedMLP(nn.Module):
    def __init__(self, LAYERS, HIDDEN_DIM, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        num_layers_per_gpu = LAYERS // world_size
        layers = []
        for _ in range(num_layers_per_gpu):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.ReLU())
        if self.rank == self.world_size - 1:
            layers.append(nn.Linear(HIDDEN_DIM, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
#--------------------------------------------------------------------------------------------------------------------------
