import torch
class Layer_Norm(torch.nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = 6e-8
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim = True)
        var = x.var(dim = -1,keepdim = True)
        out = (x - mean)/torch.sqrt(var + self.eps)
        out = out * self.scale + self.shift

        return out