'''
1. receive input or receive from previous stage if not first stage
2. forward batch through the model
3. send output to next stage if not last stage
4. 

'''
from comms import PipelineComms
from shardedmlp import ShardedMLP

def naive_pipeline_step(HIDDEN_DIM, FIXED_INPUT, FIXED_TARGET, comms:PipelineComms, model: ShardedMLP, criterion, device):
    if comms.rank == 0: ## if rank provided by communications is zero
        ### take the input
        input_data = FIXED_INPUT
    else: ## since we shape
        shape = (FIXED_INPUT, HIDDEN_DIM)
        input_data = comms.recv_forward(shape, device)
        input_data.requires_grad = True

    ## now forward pass through 
    output = model(input_data)

    if model.rank != model.world_size - 1:
        comms.send_forward(output.detach())

    if model.rank == comms.world_size - 1:
        loss = criterion(output, FIXED_TARGET)
        loss.backward()
    else:
        grad_from_next = comms.recv_backward(output.shape, device)
        output.backward(grad_from_next)
    if model.rank != 0:
        comms.send_backward(input_data.grad)

    if model.rank == comms.world_size - 1:
        return loss
        
    

        

