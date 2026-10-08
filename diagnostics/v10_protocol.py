"""Pure CPU constants for the single authorized SLAKE V10 experiment."""
EXPERIMENT = "slake_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44"
METHOD = "visual_selection_weighted_map_metanet_v10"
CONFIG_NAME = "v10_weighted_map_metanet_config.json"
WEIGHTS_NAME = "v10_weighted_map_metanet.pt"
EXPECTED_GROUP_COUNTS = {
    "p20": 51200, "visual_s8": 8192, "visual_av10": 10240,
    "question_context": 345088, "maps": 448896, "layer_condition": 387,
    "meta_net": 821920,
}
EXPECTED_TRAINABLE = 1685923
GROUP_LRS = {name: (0.3 if name == "p20" else 3e-5 if name == "visual_s8" else 1e-4)
             for name in EXPECTED_GROUP_COUNTS}
COCOOP_EVAL = "slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_143843_694315"
COCOOP_TRAIN = "slake/outputs/cocoop/slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44_20261008_125232_604532"
V1_RUN = "slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928"
