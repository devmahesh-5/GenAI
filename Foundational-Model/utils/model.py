import torch 
from .transformer import Transformer
from .layer_norm import Layer_Norm
class GPT_Model(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = torch.nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = torch.nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.dropout = torch.nn.Dropout(cfg['drop_rate'])
        self.tranformer = torch.nn.Sequential(
            *[
               Transformer(cfg) for _ in range(self.cfg['n_layers'])
            ]
        )
        self.out_norm = Layer_Norm(cfg['emb_dim'])
        self.out_head = torch.nn.Linear(cfg['emb_dim'],cfg['vocab_size'],bias=False)


    def forward(self,inp_idx):
        batch,seq_len = inp_idx.shape#batch means no of seq and seqlen means no of tokens in each seq
        x = self.tok_emb(inp_idx)
        x_pos = self.pos_emb(torch.arange(seq_len,device=inp_idx.device))
        x = x+x_pos
        x = self.dropout(x)
        x = self.tranformer(x)
        x = self.out_norm(x)
        logits = self.out_head(x)
        return logits