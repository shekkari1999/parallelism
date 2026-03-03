from comms import PipelineComms, init_distributed
import torch

def ping_pong():
    rank,world_size, device = init_distributed()
    print(world_size, rank, device)
    communications = PipelineComms(rank, world_size)
    torch.distributed.barrier()
    if rank == 0:
        tensor = torch.rand(3)
        communications.send_forward(tensor)
        print(f'sent tensor: {tensor}')
    if rank == 1:
        tensor = communications.recv_forward(3, device)
        print(f'received tensor: {tensor}')

ping_pong()