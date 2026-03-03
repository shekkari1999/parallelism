'''
Here, you will implement a basic 2 parts of MLP(though it runs on a single device).
Focus is on manual hand off of activations to the next model. All the code is self contained file.

Important takeaways shoule be:
1. activations should be passed on to the second GPU in the device format.
2. activations is derived from parameters having gradient,
so this will not break while backward.

'''
import torch
import torch.nn as nn
import torch.optim as optim
import time
#----------------------------------------------------------------------------------------------------------------------------
## some params
LAYERS = 16
HIDDEN_DIM = 128
SEED = 42
BATCH_SIZE = 32
STEPS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
## data generation also changes according to device
#----------------------------------------------------------------------------------------------------------------------------
## This will handle layers from 0 - 8
class MLP_part_1(nn.Module):
    def __init__(self, LAYERS, HIDDEN_DIM):
        super().__init__()
        layers = []
        for _ in range(LAYERS//2):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
## This will take care of 9 - 16
class MLP_part_2(nn.Module):
    def __init__(self, LAYERS, HIDDEN_DIM):
        super().__init__()
        layers = []
        for _ in range(LAYERS//2):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(HIDDEN_DIM, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
#----------------------------------------------------------------------------------------------------------------------------
## lets initialize models according to their devices
torch.manual_seed(SEED) ## set before any data creation
cuda = False
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    cuda = True
if cuda:
    mlp1 = MLP_part_1(LAYERS, HIDDEN_DIM).to(torch.cuda.device(0))
    mlp2 = MLP_part_2(LAYERS, HIDDEN_DIM).to(torch.cuda.device(1))
else:
    mlp1 = MLP_part_1(LAYERS, HIDDEN_DIM)
    mlp2 = MLP_part_2(LAYERS, HIDDEN_DIM)
#----------------------------------------------------------------------------------------------------------------------------
## declare data here
if cuda: 
    FIXED_INPUT  = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(torch.cuda.device(0))
    FIXED_TARGET = torch.randint(0, 2, (BATCH_SIZE, )).to(torch.cuda.device(1))
else:
    FIXED_INPUT  = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    FIXED_TARGET = torch.randint(0, 2, (BATCH_SIZE, ))
#----------------------------------------------------------------------------------------------------------------------------
## training loop
criterion = nn.CrossEntropyLoss()
## track optimizers for both the GPUs
optimizer = optim.Adam(list(mlp1.parameters()) + list(mlp2.parameters()), lr = 0.001)
start_time = time.time()
## start training
mlp1.train()
mlp2.train()
for step in range(STEPS):
    optimizer.zero_grad()
    activations = mlp1(FIXED_INPUT)
    if cuda:
        activations = activations.to(torch.cuda.device(1))
    #activations = activations.detach()
    logits = mlp2(activations)
    #if cuda:
    #    logits = logits.to(torch.cuda.device(1))
    #logits.retain_grad()
    loss = criterion(logits, FIXED_TARGET)
    loss.backward()
    optimizer.step()
    if step % 5 == 0:
        print(f'step: {step} | loss: {loss.item()}')
duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} Time: {duration:.3f}s")


