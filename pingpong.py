from comms import PipelineComms, init_distributed

def ping_pong():
    rank,world_size, device = init_distributed()
    print(world_size, rank, device)

ping_pong()