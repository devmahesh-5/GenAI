import numpy as np
import torch
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.tranformer[b].attn.W_query.weight = assign(
            gpt.tranformer[b].attn.W_query.weight, q_w.T)
        gpt.tranformer[b].attn.W_key.weight = assign(
            gpt.tranformer[b].attn.W_key.weight, k_w.T)
        gpt.tranformer[b].attn.W_value.weight = assign(
            gpt.tranformer[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.tranformer[b].attn.W_query.bias = assign(
            gpt.tranformer[b].attn.W_query.bias, q_b)
        gpt.tranformer[b].attn.W_key.bias = assign(
            gpt.tranformer[b].attn.W_key.bias, k_b)
        gpt.tranformer[b].attn.W_value.bias = assign(
            gpt.tranformer[b].attn.W_value.bias, v_b)

        gpt.tranformer[b].attn.out_proj.weight = assign(
            gpt.tranformer[b].attn.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.tranformer[b].attn.out_proj.bias = assign(
            gpt.tranformer[b].attn.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.tranformer[b].ff.layer[0].weight = assign(
            gpt.tranformer[b].ff.layer[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.tranformer[b].ff.layer[0].bias = assign(
            gpt.tranformer[b].ff.layer[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.tranformer[b].ff.layer[2].weight = assign(
            gpt.tranformer[b].ff.layer[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.tranformer[b].ff.layer[2].bias = assign(
            gpt.tranformer[b].ff.layer[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.tranformer[b].norm_1.scale = assign(
            gpt.tranformer[b].norm_1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.tranformer[b].norm_1.shift = assign(
            gpt.tranformer[b].norm_1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.tranformer[b].norm_2.scale = assign(
            gpt.tranformer[b].norm_2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.tranformer[b].norm_2.shift = assign(
            gpt.tranformer[b].norm_2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.out_norm.scale = assign(gpt.out_norm.scale, params["g"])
    gpt.out_norm.shift = assign(gpt.out_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

