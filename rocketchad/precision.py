import torch

def training_forward(model, precision='fp32', *, compact=False):
    if precision == 'fp32':
        return model
    if precision != 'bf16':
        raise ValueError
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError

    def forward(obs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(obs, compact_logits=True) if compact else model(obs)
        return (out[0].float(), out[1].float())
    forward.__floor_source__ = model
    return forward
