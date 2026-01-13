import torch
from .multihead_attn import MultiHeadAttention
from .ff import Feed_Forward
from .layer_norm import Layer_Norm
class Transformer(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],#this will be the context vector dim for each token
            context_length= cfg['context_length'],#total no of tokens in the input sequence
            dropout=cfg['drop_rate'],
            num_heads= cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = Feed_Forward(cfg)
        self.norm_1 = Layer_Norm(cfg['emb_dim'])
        self.norm_2 = Layer_Norm(cfg['emb_dim'])
        self.dropout_shortcut = torch.nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
        shortcut = x
        x = self.norm_1(x)
        x = self.attn(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut
        # now ff
        shortcut = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x+shortcut

        return x
        
        
    