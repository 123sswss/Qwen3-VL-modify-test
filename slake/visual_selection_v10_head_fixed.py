"""Independent V10 condition-head variant; map/native/P20 paths inherited unchanged."""
import json
from pathlib import Path
import torch
from torch import nn

from diagnostics.v10_head_fixed_protocol import METHOD, CONFIG_NAME, WEIGHTS_NAME, GROUP_COUNTS, GROUP_LRS
from slake.visual_selection_v10 import VisualSelectionV10Model


class VisualSelectionV10HeadFixedModel(VisualSelectionV10Model):
    method_name = METHOD
    group_learning_rates = GROUP_LRS

    def __init__(self, base_model, init_seed=44):
        super().__init__(base_model,init_seed=init_seed)
        shared = {name:p.detach().clone() for name,p in self.named_parameters()
                  if p.requires_grad and not name.startswith("meta_net.2.")}
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(init_seed).get_state())
            # Same private initialization sequence as V10, but no final override.
            self.meta_net = nn.Sequential(nn.Linear(2560,160),nn.ReLU(),nn.Linear(160,2560))
            self.summary_norm = nn.LayerNorm(2560)
        self.meta_net.to(device=self.p20.device,dtype=torch.float32)
        self.summary_norm.to(device=self.p20.device,dtype=torch.float32)
        final = dict(self.named_parameters())
        if any(not torch.equal(value,final[name]) for name,value in shared.items()):
            raise RuntimeError("Retained V10 initialization changed")
        self.initialization_audit.update(
            reference="same_seed_original_V10",shared_tensor_count=len(shared),
            all_shared_initial_values_equal=True,meta_first_init="nn.Linear_default",
            meta_output_init="nn.Linear_default",meta_output_bias_zero=False,
            meta_output_actual_std=float(self.meta_net[2].weight.detach().std()),
            meta_output_bias_rms=float(self.meta_net[2].bias.detach().square().mean().sqrt()),
            summary_norm_init="LayerNorm_default_weight1_bias0_eps1e-5")
        for handle in self._first_grad_handles:
            handle.remove()
        self._first_grad_handles = []
        self.first_backward_gradients = {}
        self._install_first_backward_hooks()
        self._audit_parameters()
        print("[V10_HEAD_FIXED_INIT] "+json.dumps(self.initialization_audit),flush=True)

    def trainable_parameter_groups(self):
        groups = super().trainable_parameter_groups()
        if hasattr(self,"summary_norm"):
            groups["summary_norm"] = list(self.summary_norm.parameters())
        return groups

    def _audit_parameters(self):
        if not hasattr(self,"summary_norm"):
            return super()._audit_parameters()
        groups = self.trainable_parameter_groups()
        grouped = [p for values in groups.values() for p in values]
        counts = {name:sum(p.numel() for p in values) for name,values in groups.items()}
        if counts != GROUP_COUNTS or len(grouped) != len({id(p) for p in grouped}) or {
                id(p) for p in grouped} != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError(f"Head-fixed parameter groups differ: {counts}")
        print("[V10_HEAD_FIXED_PARAMETERS] "+json.dumps(counts),flush=True)
        return counts

    def _prefix_shift(self, condition):
        normalized = self.summary_norm(condition)
        self._prefix_shift_debug = {
            "summary_pre_ln_rms":condition.float().square().mean().sqrt().detach(),
            "summary_post_ln_rms":normalized.float().square().mean().sqrt().detach(),
        }
        return self.meta_net(normalized)

    def save_v10(self, directory):
        path = Path(directory)
        path.mkdir(parents=True,exist_ok=True)
        config = {"method":METHOD,"init_seed":self.init_seed,"layers":[5,11,17],
                  "visual_tokens":[8,10],"prefix_tokens":20,"summary_norm":2560,
                  "meta_net":[2560,160,2560],"meta_init":"nn.Linear_default_both_layers",
                  "trainable_parameters":self._audit_parameters()}
        (path/CONFIG_NAME).write_text(json.dumps(config,indent=2),encoding="utf-8")
        torch.save({name:p.detach().cpu() for name,p in self.named_parameters() if p.requires_grad},path/WEIGHTS_NAME)

    def load_v10(self, directory):
        path = Path(directory)
        config = json.loads((path/CONFIG_NAME).read_text(encoding="utf-8"))
        if config["method"] != METHOD or config["init_seed"] != self.init_seed or config["trainable_parameters"] != GROUP_COUNTS:
            raise ValueError("Not the matching independent condition-head checkpoint")
        state = torch.load(path/WEIGHTS_NAME,map_location="cpu",weights_only=True)
        parameters = {name:p for name,p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Condition-head checkpoint parameter set differs")
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(state[name].to(p.device))
