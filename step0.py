'''
Here, you will implement a basic MLP, that will take a fixed batch as input, 
calculates loss, updates gradients for 50 steps. Everything runs in one model.

'''
import torch
import torch.nn as nn
import torch.optim as optim
import time

## some params
LAYERS = 16
HIDDEN_DIM = 128
SEED = 42
BATCH_SIZE = 32
STEPS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#--------------------------------------------------------------------------------------------------------------------------
## lets define a basic neural network
class MLP(nn.Module):
    def __init__(self, HIDDEN_DIM, LAYERS):
        super().__init__()
        layers = []
        for _ in range(LAYERS):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(HIDDEN_DIM, 2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
#----------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(SEED) ## set before any data creation
## initalize our MLP
mlp = MLP(HIDDEN_DIM, LAYERS)
MODEL = mlp
print(f'Total number of model parameters are {sum(p.numel() for p in MODEL.parameters())}')
param_memory_bytes = sum(p.numel() * p.element_size() for p in MODEL.parameters())
param_memory_mb = param_memory_bytes / (1024 ** 2)
print(f"Total memory occupied by these model parameters are: {param_memory_mb:.4f} MB")
#print(mlp)
#----------------------------------------------------------------------------------------------------------------------------
FIXED_INPUT = torch.randn(BATCH_SIZE, HIDDEN_DIM)
FIXED_TARGET = torch.randint(0, 2, (BATCH_SIZE,)) ## This is our targets 
#----------------------------------------------------------------------------------------------------------------------------

## training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(MODEL.parameters(), lr = 0.001)
start_time = time.time()
MODEL.train()
for step in range(STEPS):
    optimizer.zero_grad()
    logits = MODEL(FIXED_INPUT)
    loss = criterion(logits, FIXED_TARGET)
    loss.backward()
    optimizer.step()
    if step % 5 == 0:
        print(f'step: {step} | loss: {loss.item()}')
duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} Time: {duration:.3f}s")
