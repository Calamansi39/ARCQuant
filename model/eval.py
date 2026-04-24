import torch
import torch.nn as nn
from tqdm import tqdm
import fnmatch


def _build_position_embeddings(model, hidden_states, position_ids):
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None:
        return None
    if position_ids is None:
        position_ids = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)
    return rotary_emb(hidden_states, position_ids)


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

@torch.no_grad()
def eval_ppl(model, testenc, dev):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // 2048
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
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
    model.to("cuda:0")
    for i in range(nsamples):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(dev)
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
        layer = layers[i].to(dev)
        for j in range(nsamples):
            hidden_states = inps[j].unsqueeze(0)
            layer_kwargs = dict(attention_mask=attention_mask, position_ids=position_ids)
            position_embeddings = _build_position_embeddings(model, hidden_states, position_ids)
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            outs[j] = layer(hidden_states, **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
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
