import torch
import torch.nn as nn
from shardedmlp import ShardedMLP
from comms import PipelineComms, init_distributed
import torch.optim as optim
import time
from naive_pipeline_parallel import naive_pipeline_step

## some params
LAYERS = 16
HIDDEN_DIM = 128
SEED = 42
BATCH_SIZE = 32
STEPS = 50
#--------------------------------------------------------------------------------------------------------------------------
## STEP 1: Setting up distributed environment

rank, world_size, device = init_distributed()
comms = PipelineComms(rank, world_size)
#--------------------------------------------------------------------------------------------------------------------------
## take care of reproducability
torch.manual_seed(SEED)
for i in range(2 * rank * (LAYERS // world_size)):
    torch.randn(1)
#--------------------------------------------------------------------------------------------------------------------------
## STEP 2: Initialize model
model = ShardedMLP(LAYERS, HIDDEN_DIM, rank, world_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.001)
## cuda stuff(and also only rank 0 loads the data)
if rank == 0:
    FIXED_INPUT = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(device) ## device is being returned by process
else: 
    FIXED_INPUT = BATCH_SIZE

## only last rank has targets
if rank == world_size - 1:
    FIXED_TARGET = torch.randint(0, 2, (BATCH_SIZE, ))
else:
    FIXED_TARGET = None
#--------------------------------------------------------------------------------------------------------------------------
## training loop
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    start_time = time.time()
    ### if you are last gpu, calculate and return loss, else just perform a training step
    if rank == world_size - 1:
        loss = naive_pipeline_step(HIDDEN_DIM, FIXED_INPUT, FIXED_TARGET, comms,model, criterion, device)
    else:
        naive_pipeline_step(HIDDEN_DIM, FIXED_INPUT, FIXED_TARGET,  comms, model, criterion, device)
    optimizer.step()
    if rank == world_size - 1 and step % 5 == 0:
        print(f'step: {step} | loss: {loss.item()}')
#--------------------------------------------------------------------------------------------------------------------------
## clean up
if rank == world_size - 1:
    print('----Training Complete----')
    duration = time.time() - start_time
    print(f'final loss: {loss.item(): .6f} | Time: {duration: .3f}')
torch.distributed.destroy_process_group()



    







