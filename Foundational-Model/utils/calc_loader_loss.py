import torch
def calc_batch_loss(input_batch,target_batch,model,device):
    input_batch,target_batch = input_batch.to(device),target_batch.to(device)
    logits = model(input_batch)#for a batch there can have multiple input sequence of inputs which are sent to model class and we get input in batch_size*no_of_tokens_in_one_sequence*vocab_size
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())#target was 2*256 before flatten
    return loss

def calc_loader_loss(data_loader,model,device,num_batches=None):
    train_loss =0 
    if(len(data_loader)==0):
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)#it gives no of batches in a dataloader where in each batch there are multiple input and target sequences
        # we will have avg loss of these input sequences among different loss
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_batch_loss(input_batch=input_batch,target_batch=target_batch,model=model,device=device)
            #this batch are the one from one batch which are of dim 2*256 which will later in model will be done a vector embedding of dim 768 for each token to get each batch have dimension of 2*256*768 and later inside same model it will return 2*256*50257 returned as logits and these will be performed softmax where each toke have vector for every vocabloury  and then highest probablity token is chosen as output for each input token 
            train_loss +=loss

        else:
            break

    return train_loss/num_batches