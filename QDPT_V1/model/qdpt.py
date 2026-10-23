import weakref
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

LAYERS = (5, 11, 17)


class Visual18(nn.Module):
    def __init__(self, block, owner):
        super().__init__()
        self.block = block
        self.owner = weakref.ref(owner)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, **kwargs):
        owner = self.owner()
        # [b+1] -> [b]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        # [8,1024] + [10,1024] -> [18,1024]
        prompt = torch.cat((owner.visual_s8, owner.visual_av10)).to(hidden_states.dtype)
        # [Σn,1024] -> [Σ(n+18),1024]
        hidden_states = torch.cat([
            torch.cat((prompt, chunk)) for chunk in hidden_states.split(lengths)
        ])

        def prepend(tensor, value):
            # [Σn,d] -> [Σ(n+18),d]
            return torch.cat([
                torch.cat((chunk.new_full((18, *chunk.shape[1:]), value), chunk))
                for chunk in tensor.split(lengths)
            ])

        if rotary_pos_emb is not None:
            rotary_pos_emb = prepend(rotary_pos_emb, 0)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # [18,d]: cos=1, sin=0
            position_embeddings = prepend(cos, 1), prepend(sin, 0)
        if 'max_seqlen' in kwargs:
            kwargs['max_seqlen'] += 18
        cu = cu_seqlens + torch.arange(cu_seqlens.numel(), device=cu_seqlens.device) * 18
        output = self.block(hidden_states, cu.to(cu_seqlens.dtype), rotary_pos_emb,
                            position_embeddings, **kwargs)
        # [Σ(n+18),1024] -> [Σn,1024]
        return torch.cat([chunk[18:] for chunk in output.split([n + 18 for n in lengths])])


class QDPT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model.requires_grad_(False)
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        embedding = base_model.get_input_embeddings().weight
        rows = torch.randint(len(embedding), (20,), generator=torch.Generator().manual_seed(44))
        # [vocab,2560] -> [20,2560]
        self.p20 = nn.Parameter(embedding.detach()[rows.to(embedding.device)].float().clone())
        self.visual_s8 = nn.Parameter(torch.empty(8, 1024, device=embedding.device))
        self.visual_av10 = nn.Parameter(torch.empty(10, 1024, device=embedding.device))
        nn.init.normal_(self.visual_s8, std=.02)
        nn.init.normal_(self.visual_av10, std=.02)
        self.text_projection = nn.Linear(2560, 128, bias=False)
        self.question_depthwise = nn.Conv1d(128, 128, 3, padding=1, groups=128)
        self.question_pointwise = nn.Conv1d(128, 128, 1)
        self.question_norm = nn.LayerNorm(128)
        self.question_pool = nn.Linear(128, 1, bias=False)
        self.query_heads = nn.ModuleList(nn.Linear(128, 128) for _ in LAYERS)
        self.key_norms = nn.ModuleList(nn.LayerNorm(1024) for _ in LAYERS)
        self.key_heads = nn.ModuleList(nn.Linear(1024, 128, bias=False) for _ in LAYERS)
        self.layer_gate = nn.Linear(128, 3)
        self.value_norms = nn.ModuleList(nn.LayerNorm(2560) for _ in LAYERS)
        self.value_blocks = nn.ModuleList(nn.Linear(2560, 64, bias=False) for _ in LAYERS)
        self.prefix_output = nn.Linear(192, 2560)
        nn.init.normal_(self.prefix_output.weight, std=1e-4)
        nn.init.zeros_(self.prefix_output.bias)
        for name, module in self.named_children():
            if name != 'base_model':
                module.to(device=embedding.device, dtype=torch.float32)
        visual = base_model.model.visual
        visual.blocks[17] = Visual18(visual.blocks[17], self)
        self.features = {}
        for layer in LAYERS:
            visual.blocks[layer].register_forward_hook(self.capture(layer))
        visual.merger.register_forward_hook(self.capture('value'))

    def capture(self, key):
        def hook(module, args, output):
            self.features[key] = output
        return hook

    def condition(self, question_ids, valid, grid):
        # [b,q] -> [b,q,2560]
        question = self.base_model.get_input_embeddings().weight[question_ids].float()
        # [b,q,2560] -> [b,q,128]
        x = self.text_projection(question) * valid.unsqueeze(-1)
        # [b,q,128] -> [b,128,q]
        conv = self.question_pointwise(F.gelu(self.question_depthwise(x.transpose(1, 2))))
        # [b,128,q] -> [b,q,128]
        x = (x + conv.transpose(1, 2)) * valid.unsqueeze(-1)
        # [b,q,128] -> [b,q]
        scores = self.question_pool(self.question_norm(x)).squeeze(-1)
        # [b,q,1] * [b,q,128] -> [b,128]
        u = (scores.masked_fill(~valid, -torch.inf).softmax(-1).unsqueeze(-1) * x).sum(1)
        # [b,128] -> [b,3]
        beta = self.layer_gate(u).softmax(-1)
        # [b,3] -> [n₀,...,nᵦ₋₁], nᵢ=tᵢ*hᵢ*wᵢ
        counts = [t * h * w for t, h, w in grid.tolist()]
        # [Σ(n/4),2560] -> b * [n/4,2560]
        values = self.features['value'].float().split([n // 4 for n in counts])
        blocks = []
        for i, layer in enumerate(LAYERS):
            # [b,128]
            query = self.query_heads[i](u)
            summaries = []
            for b, (patches, value) in enumerate(zip(self.features[layer].float().split(counts), values)):
                # [n,1024] -> [n,128]
                keys = self.key_heads[i](self.key_norms[i](patches))
                # [n,128] * [128] -> [n]
                probabilities = ((keys * query[b]).sum(-1) / 128**.5).softmax(0)
                # [n] -> [n/4,4] -> [n/4]
                merged = probabilities.reshape(-1, 4).sum(1)
                # [n/4,1] * [n/4,2560] -> [2560]
                summaries.append((merged[:, None] * value).sum(0))
            # b * [2560] -> [b,2560] -> [b,64]
            z = self.value_blocks[i](self.value_norms[i](torch.stack(summaries)))
            # [b,64] * [b,1]
            blocks.append(z * beta[:, i, None])
        # 3 * [b,64] -> [b,192] -> [b,2560]
        return self.prefix_output(F.relu(torch.cat(blocks, -1)))

    @contextmanager
    def inject(self, question_ids, valid, grid):
        done = False

        def static_prefix(module, args, output):
            if done:
                return output
            # [20,2560] -> [b,20,2560]
            prefix = self.p20.to(output.dtype)[None].expand(output.shape[0], -1, -1)
            # [b,20,2560] + [b,t,2560] -> [b,t+20,2560]
            return torch.cat((prefix, output[:, 20:]), 1)

        def conditional_prefix(module, args, kwargs):
            nonlocal done
            if not done:
                # [b,t+20,2560]
                x = kwargs['inputs_embeds']
                # [b,2560]
                shift = self.condition(question_ids, valid, grid).to(x.dtype)
                # [b,20,2560] + [b,1,2560]
                kwargs['inputs_embeds'] = torch.cat((x[:, :20] + shift[:, None], x[:, 20:]), 1)
                done = True
            return args, kwargs

        embed_hook = self.base_model.get_input_embeddings().register_forward_hook(static_prefix)
        llm_hook = self.base_model.model.language_model.register_forward_pre_hook(
            conditional_prefix, with_kwargs=True)
        try:
            yield
        finally:
            embed_hook.remove()
            llm_hook.remove()
            self.features.clear()

    def call(self, generate, batch):
        # [b,q], [b,q]
        question_ids = batch.pop('question_source_ids')
        valid = batch.pop('question_source_mask')
        ids = batch['input_ids']
        # [b,20]
        prefix = ids.new_zeros((ids.shape[0], 20))
        # [b,t] -> [b,t+20]
        batch['input_ids'] = torch.cat((prefix, ids), 1)
        batch['attention_mask'] = torch.cat((torch.ones_like(prefix), batch['attention_mask']), 1)
        if 'labels' in batch:
            # [b,20]=-100; [b,t+20]
            batch['labels'] = torch.cat((prefix.new_full(prefix.shape, -100), batch['labels']), 1)
        with self.inject(question_ids, valid, batch['image_grid_thw']):
            fn = self.base_model.generate if generate else self.base_model
            return fn(**batch)

    def forward(self, **batch):
        return self.call(False, batch)

    def generate(self, **batch):
        return self.call(True, batch)

    def optimizer_groups(self):
        def parameters(*modules):
            return [p for module in modules for p in module.parameters()]
        groups = {
            'p20': (.3, [self.p20]),
            'question_context': (1e-4, parameters(self.text_projection, self.question_depthwise,
                self.question_pointwise, self.question_norm, self.question_pool)),
            'maps': (1e-4, parameters(self.query_heads, self.key_norms, self.key_heads)),
            'layer_condition': (1e-4, parameters(self.layer_gate, self.value_norms)),
            'prefix_output': (3e-4, parameters(self.prefix_output)),
            'visual_s8': (3e-5, [self.visual_s8]),
            'visual_av10': (1e-4, [self.visual_av10]),
            'value_projection': (3e-4, parameters(self.value_blocks)),
        }
        return [dict(params=p, lr=lr, weight_decay=0., name=name) for name, (lr, p) in groups.items()]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad},
                   directory / 'visual_selection_prefix.pt')

    def load(self, directory):
        weights = torch.load(Path(directory) / 'visual_selection_prefix.pt', map_location='cpu', weights_only=True)
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(weights[name].to(parameter.device))
