import warnings
from functools import partial

import torch
import torch.nn as nn
from loguru import logger

from romatch.models.encoders import VIT_BACKBONES, CNNandDinov2, CNNandViT
from romatch.models.matcher import (
    GP,
    ConvRefiner,
    CosKernel,
    Decoder,
    RegressionMatcher,
)
from romatch.models.fewshot import FewShotRegressionMatcher
from romatch.models.tiny import TinyRoMa
from romatch.models.transformer import Block, MemEffAttention, TransformerDecoder


def tiny_roma_v1_model(
    weights=None, freeze_xfeat=False, exact_softmax=False, xfeat=None
):
    model = TinyRoMa(
        xfeat=xfeat, freeze_xfeat=freeze_xfeat, exact_softmax=exact_softmax
    )
    if weights is not None:
        model.load_state_dict(weights)
    return model

def pad_refiner_state_dict(state_dict_old,state_dict_pad):
    for key in state_dict_pad.keys():
        if key.startswith('decoder.conv_refiner'):
            param = state_dict_old[key]
            shape_old = param.shape
            shape_pad = state_dict_pad[key].shape
            if shape_old != shape_pad:
                new_param = torch.zeros(shape_pad, device=param.device, dtype=param.dtype)
                slices = tuple(slice(0, s) for s in shape_old)
                new_param[slices] = param
                state_dict_old[key] = new_param
    return state_dict_old

def roma_model_pad(
    resolution,
    upsample_preds,
    device=None,
    weights=None,
    dinov2_weights=None,
    amp_dtype: torch.dtype = torch.float16,
    use_custom_corr=True,
    symmetric=True,
    upsample_res=None,
    sample_thresh=0.05,
    sample_mode="threshold_balanced",
    attenuate_cert = True,
    refiner_channels= [1384, 1144, 576, 144, 24],
    **kwargs,
):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    if str(device) == "cpu":
        amp_dtype = torch.float32

    assert resolution[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert resolution[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    logger.info(
        f"Using coarse resolution {resolution}, and upsample res {upsample_res}"
    )

    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    partial_conv_refiner = partial(
        ConvRefiner,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        corr_in_other=True,
        amp=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
        use_custom_corr=use_custom_corr,
    )

    conv_refiner = nn.ModuleDict(
        {
            "16": partial_conv_refiner(
                refiner_channels[0],
                refiner_channels[0],
                2 + 1,
                displacement_emb_dim=128,
                local_corr_radius=7,
            ),
            "8": partial_conv_refiner(
                refiner_channels[1],
                refiner_channels[1],
                2 + 1,
                displacement_emb_dim=64,
                local_corr_radius=3,
            ),
            "4": partial_conv_refiner(
                refiner_channels[2],
                refiner_channels[2],
                2 + 1,
                displacement_emb_dim=32,
                local_corr_radius=2,
            ),
            "2": partial_conv_refiner(
                refiner_channels[3],
                refiner_channels[3],
                2 + 1,
                displacement_emb_dim=16,
            ),
            "1": partial_conv_refiner(
                refiner_channels[4],
                refiner_channels[4],
                2 + 1,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=False, amp=True),
        amp=True,
        dinov2_weights=dinov2_weights,
        amp_dtype=amp_dtype,
    )
    h, w = resolution
    
    matcher = RegressionMatcher(
        encoder,
        decoder,
        h=h,
        w=w,
        upsample_preds=upsample_preds,
        upsample_res=upsample_res,
        symmetric=symmetric,
        attenuate_cert=attenuate_cert,
        sample_mode=sample_mode,
        sample_thresh=sample_thresh,
        **kwargs,
    ).to(device)
    if weights is not None:
        state_dict_pad = matcher.state_dict()
        weights = pad_refiner_state_dict(weights,state_dict_pad)
        del state_dict_pad

    matcher.load_state_dict(weights)
    return matcher


def roma_model(
    resolution,
    upsample_preds,
    device=None,
    weights=None,
    dinov2_weights=None,
    amp_dtype: torch.dtype = torch.float16,
    use_custom_corr=True,
    symmetric=True,
    upsample_res=None,
    sample_thresh=0.05,
    sample_mode="threshold_balanced",
    attenuate_cert = True,
    **kwargs,
):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    if str(device) == "cpu":
        amp_dtype = torch.float32

    assert resolution[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert resolution[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    logger.info(
        f"Using coarse resolution {resolution}, and upsample res {upsample_res}"
    )

    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    partial_conv_refiner = partial(
        ConvRefiner,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        corr_in_other=True,
        amp=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
        use_custom_corr=use_custom_corr,
    )

    conv_refiner = nn.ModuleDict(
        {
            "16": partial_conv_refiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=128,
                local_corr_radius=7,
            ),
            "8": partial_conv_refiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=64,
                local_corr_radius=3,
            ),
            "4": partial_conv_refiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=32,
                local_corr_radius=2,
            ),
            "2": partial_conv_refiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                displacement_emb_dim=16,
            ),
            "1": partial_conv_refiner(
                2 * 9 + 6,
                24,
                2 + 1,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=False, amp=True),
        amp=True,
        dinov2_weights=dinov2_weights,
        amp_dtype=amp_dtype,
    )
    h, w = resolution
    
    matcher = RegressionMatcher(
        encoder,
        decoder,
        h=h,
        w=w,
        upsample_preds=upsample_preds,
        upsample_res=upsample_res,
        symmetric=symmetric,
        attenuate_cert=attenuate_cert,
        sample_mode=sample_mode,
        sample_thresh=sample_thresh,
        **kwargs,
    ).to(device)
    matcher.load_state_dict(weights)
    return matcher


def roma_fewshot_model(
    resolution,
    upsample_preds,
    device=None,
    weights=None,
    dinov2_weights=None,
    amp_dtype: torch.dtype = torch.float16,
    use_custom_corr=True,
    symmetric=True,
    upsample_res=None,
    sample_thresh=0.05,
    sample_mode="threshold_balanced",
    attenuate_cert=True,
    align_scales=(16,),
    align_kwargs=None,
    backbone="dinov2",
    backbone_tuning="frozen",
    lora_rank=8,
    lora_alpha=16.0,
    **kwargs,
):
    """Few-shot registration matcher: same graph as ``roma_model`` but wrapped in a
    ``FewShotRegressionMatcher`` with per-scale ``SupportAlignmentModule``s.

    ``weights`` (official RoMa outdoor state dict) is loaded with ``strict=False`` so the
    only missing keys are the freshly-initialised alignment params. Passing
    ``weights=None`` skips loading (useful for the identity smoke test).

    ``backbone`` / ``backbone_tuning`` select the scale-16 ViT and what trains in it; see
    :mod:`romatch.models.encoders`. The default pair is RoMa's own frozen DINOv2, so an
    existing command line builds the same graph it always did. A non-default backbone
    changes what the coarse branch emits, so the decoder half of ``weights`` is then a
    warm start rather than a matched one, and the arm has to be trained.
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    if str(device) == "cpu":
        amp_dtype = torch.float32

    spec = VIT_BACKBONES[backbone]
    patch, coarse_dim = spec["patch_size"], spec["embed_dim"]
    assert resolution[0] % patch == 0, f"Needs to be multiple of {patch} for {backbone}"
    assert resolution[1] % patch == 0, f"Needs to be multiple of {patch} for {backbone}"

    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    partial_conv_refiner = partial(
        ConvRefiner,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        corr_in_other=True,
        amp=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
        use_custom_corr=use_custom_corr,
    )
    conv_refiner = nn.ModuleDict(
        {
            "16": partial_conv_refiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=128,
                local_corr_radius=7,
            ),
            "8": partial_conv_refiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=64,
                local_corr_radius=3,
            ),
            "4": partial_conv_refiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                displacement_emb_dim=32,
                local_corr_radius=2,
            ),
            "2": partial_conv_refiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                displacement_emb_dim=16,
            ),
            "1": partial_conv_refiner(
                2 * 9 + 6,
                24,
                2 + 1,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(coarse_dim, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict(
        {"16": proj16, "8": proj8, "4": proj4, "2": proj2, "1": proj1}
    )
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
    )

    encoder = CNNandViT(
        cnn_kwargs=dict(pretrained=False, amp=True),
        amp=True,
        vit_weights=dinov2_weights,
        amp_dtype=amp_dtype,
        backbone=backbone,
        backbone_tuning=backbone_tuning,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    h, w = resolution

    matcher = FewShotRegressionMatcher(
        encoder,
        decoder,
        pyramid_dims={16: coarse_dim},
        h=h,
        w=w,
        upsample_preds=upsample_preds,
        upsample_res=upsample_res,
        symmetric=symmetric,
        attenuate_cert=attenuate_cert,
        sample_mode=sample_mode,
        sample_thresh=sample_thresh,
        align_scales=align_scales,
        align_kwargs=align_kwargs,
        **kwargs,
    ).to(device)
    if weights is not None:
        missing, unexpected = matcher.load_state_dict(weights, strict=False)
        logger.info(
            f"roma_fewshot_model: loaded weights strict=False "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
    return matcher
