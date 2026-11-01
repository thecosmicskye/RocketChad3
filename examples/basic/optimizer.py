import torch

from model import muon_parameters
from rocketchad import BatchedMuon, Optimizers


# Muon uses selected matrices; AdamW updates the remaining parameters.
# make_optimizer may return a PyTorch-style object with param_groups,
# zero_grad, step, state_dict and load_state_dict. Custom optimizers run eagerly by default;
# checkpoint state must contain tensors and basic Python types.
# Combine graph-aware optimizers with Optimizers(captured=(...), eager=(...)).
# Steps must leave gradient tensors in place; captured optimizers also need CUDA
# graph support and tensor state in .state. Use Optimizers(eager=(...), graph=False)
# for ordinary optimizers.
def make_optimizer(model, config):
    name = config['optimizer']
    if name != 'muon':
        optimizer = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[name]
        optimizer = optimizer(model.parameters(), lr=config[f'{name}_lr'], eps=config['optimizer_eps'],
                              weight_decay=config['weight_decay'])
        return Optimizers(eager=(optimizer,))

    selected = list(muon_parameters(model))
    selected_ids = {id(parameter) for parameter in selected}
    other = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in selected_ids]
    muon = BatchedMuon(selected, lr=config['muon_lr'], momentum=config['muon_momentum'], nesterov=True,
                       ns_steps=config['muon_ns_steps'], adjust_lr_fn='match_rms_adamw',
                       weight_decay=config['weight_decay'], eps=1e-7)
    muon.compute_dtype = torch.bfloat16 if config['precision'] == 'bf16' else torch.float32
    adamw = torch.optim.AdamW(other, lr=config['adamw_lr'], eps=config['optimizer_eps'],
                              weight_decay=config['weight_decay'])
    return Optimizers(captured=(muon,), eager=(adamw,))
