'''
Now lets Distributed Training basics
1. Process group: Just like lobby. (In a multiplayer game)
2. World_size: # GPUs
3. Rank: unique id of each GPU
'''

import torch
import torch.distributed as dist
import os
#----------------------------------------------------------------------------------------------------------------------------
def init_distributed():
    ### these environment variables will be set by torch run
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    ## set the device
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    ## initialize process group according to device
    if torch.cuda.is_available():
        dist.init_process_group(backend = 'nccl', rank = rank, world_size = world_size)
    else:
        dist.init_process_group(backend = 'gloo', rank = rank, world_size = world_size)
    return rank, world_size, device

class PipelineComms:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # Define Neighbors
        # If I am Rank 0, I have no previous neighbor (None)
        self.prev_rank = rank - 1 if rank > 0 else None
        # If I am the last Rank, I have no next neighbor (None)
        self.next_rank = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor):
        """Send activation to the next GPU."""
        # .contiguous() is required before sending
        dist.send(tensor.contiguous(), dst=self.next_rank)

    def recv_forward(self, shape, device, dtype=torch.float32):
        """Receive activation from the previous GPU."""
        # We must allocate an empty buffer to receive the data
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.prev_rank)
        return tensor

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        # Blocking communication (dist.send) means
        # the program waits until the send is complete
        # before proceeding, which is simple and easier
        # to reason about. Async (isend) allows overlapping
        # computation and communication,
        # increasing efficiency and complexity.
        dist.send(tensor.contiguous(), dst=self.prev_rank)

    def recv_backward(self, shape, device, dtype=torch.float32):
        """Receive gradients from the next GPU."""
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.next_rank)
        return tensor

    def isend_forward(self, tensor):
        return dist.isend(tensor.contiguous(), dst=self.next_rank)





