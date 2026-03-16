import torch
import torch.nn as nn
from tqdm import tqdm
import fnmatch

def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

@torch.no_grad()
def eval_ppl(model, testenc, dev):
    if isinstance(dev, str):
        devices = [item.strip() for item in dev.split(',') if item.strip()]
    else:
        devices = list(dev)
    if not devices:
        devices = ['cuda:0']
    primary_device = devices[0]

    testenc = testenc.input_ids
    nsamples = testenc.numel() // 2048
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(primary_device)
    layers[0] = layers[0].to(primary_device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, model.config.hidden_size), dtype=dtype, device=primary_device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    model.to(primary_device)
    for i in range(nsamples):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(primary_device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        layer_device = devices[i % len(devices)]
        layer = layers[i].to(layer_device)
        for j in range(nsamples):
            layer_attention_mask = attention_mask.to(layer_device) if attention_mask is not None else None
            layer_position_ids = position_ids.to(layer_device) if position_ids is not None else None
            layer_out = layer(
                inps[j].unsqueeze(0).to(layer_device),
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
            )[0]
            outs[j] = layer_out.to(primary_device)
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(primary_device)
    model.lm_head = model.lm_head.to(primary_device)

    testenc = testenc.to(primary_device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * 2048):((i + 1) * 2048)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

    return ppl.item()
