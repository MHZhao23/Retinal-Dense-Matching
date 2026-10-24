from typing import Union
import torch
from .roma_models import roma_model,roma_model_pad, tiny_roma_v1_model, roma_fewshot_model
from romatch.models.encoders import load_vit_weights


weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # hopefully this doesnt change :D
}


def tiny_roma_v1_outdoor(device, weights=None, xfeat=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["tiny_roma_v1"]["outdoor"], map_location=device
        )
    if xfeat is None:
        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        ).net

    return tiny_roma_v1_model(weights=weights, xfeat=xfeat).to(device)


def roma_outdoor(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
    with_padding=False,
    do_compile=False,
):
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Float32 matmul precision must be set to highest for RoMa. See also https://github.com/Parskatt/RoMaV2/issues/35")

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["outdoor"], map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    model_init = roma_model if not with_padding else roma_model_pad
    model = model_init(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
    )
    if do_compile:
        model.compile()
    return model


def roma_fewshot(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=False,
    attenuate_cert=True,
    align_scales=(16,),
    align_kwargs=None,
    load_pretrained=True,
    backbone="dinov2",
    backbone_tuning="frozen",
    backbone_weights=None,
    lora_rank=8,
    lora_alpha=16.0,
):
    """Few-shot registration matcher. Loads official RoMa outdoor weights (strict=False)
    unless ``load_pretrained=False`` (e.g. for the identity smoke test).

    ``backbone`` picks the scale-16 ViT (``dinov2`` / ``dinov3`` / ``retfound``) and
    ``backbone_tuning`` what trains in it (``frozen`` / ``unfrozen`` / ``lora``); see
    :mod:`romatch.models.encoders`. Only DINOv2 has a download URL here -- the other two
    are gated or form-walled, so they are passed in through ``backbone_weights`` as a
    path or a state dict."""
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Float32 matmul precision must be set to highest for RoMa. See also https://github.com/Parskatt/RoMaV2/issues/35")

    if weights is None and load_pretrained:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["outdoor"], map_location=device
        )
    if backbone_weights is None:
        backbone_weights = dinov2_weights  # the older name for the same argument
    if backbone_weights is None:
        if backbone != "dinov2":
            raise ValueError(
                f"backbone {backbone!r} has no download here; pass its checkpoint "
                "through backbone_weights (--backbone_weights on the trainer)"
            )
        backbone_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    backbone_weights = load_vit_weights(backbone_weights, map_location=device)
    return roma_fewshot_model(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=backbone_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
        attenuate_cert=attenuate_cert,
        align_scales=align_scales,
        align_kwargs=align_kwargs,
        backbone=backbone,
        backbone_tuning=backbone_tuning,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )


def roma_indoor(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
    with_padding=False,
    do_compile=False,
):
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Float32 matmul precision must be set to highest for RoMa. See also https://github.com/Parskatt/RoMaV2/issues/35")

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["indoor"], map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    model_init = roma_model if not with_padding else roma_model_pad
    model = model_init(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
    )
    if do_compile:
        model.compile()
    return model