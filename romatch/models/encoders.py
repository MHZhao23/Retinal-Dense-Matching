"""Feature encoders: a VGG19 pyramid plus one ViT for the coarse (scale-16) branch.

The ViT is swappable, which is the point of the backbone study: RoMa's scale-16 features
are dense *patch tokens*, so the branch takes any ViT that produces them, and the three
supported choices differ in what they were pretrained on.

=============  ==========  =====  ============================================
name           arch        patch  pretraining
=============  ==========  =====  ============================================
``dinov2``     ViT-L/14    14     LVD-142M natural images (RoMa's own, default)
``dinov3``     ViT-L/16    16     LVD-1689M natural images, dense-feature SSL
``retfound``   ViT-L/16    16     1.6M retinal images, MAE
=============  ==========  =====  ============================================

All three are ViT-L, so the branch is 1024-wide either way and ``proj16`` loads from the
official RoMa weights unchanged. The patch size does move: at 560 a /14 backbone gives a
40x40 coarse grid and a /16 one gives 35x35. Nothing downstream hardcodes it -- the
decoder reads its sizes off the tensors -- but the resolution assertions in
``roma_models`` follow ``patch_size`` rather than a literal 14.

``tuning`` decides what the optimizer sees:

``frozen``
    The default and what RoMa itself does. The ViT is kept out of the module tree
    entirely (the list-hack below), runs under ``no_grad`` in ``amp_dtype``, and appears
    in neither ``parameters()`` nor the state dict.
``unfrozen``
    The ViT is a real submodule in fp32 and is trained end to end. Its weights then
    appear in the checkpoint under ``encoder.backbone.vit.*``; official RoMa weights
    carry no such keys, so a ``strict=False`` load leaves them at their pretrained
    values, which is what is wanted.
``lora``
    As ``unfrozen``, but only the injected low-rank adapters on each block's attention
    projections carry gradient. See :class:`LoRALinear`.

CLIP-style medical models (BiomedCLIP, FLAIR, Ret-CLIP, RetiZero) are deliberately absent:
they optimise a global image-text embedding, and nothing in their objective asks the patch
tokens to stay spatially coherent, which is the one property this branch needs.
"""

import os
import timm
import torch
import torch.nn as nn
import torchvision.models as tvm
from romatch.utils.utils import get_autocast_params
from .transformer import vit_large

VIT_BACKBONES = {
    "dinov2": dict(embed_dim=1024, patch_size=14, timm_names=()),
    "dinov3": dict(embed_dim=1024, patch_size=16, fp16_overflows=True, timm_names=(
        "vit_large_patch16_dinov3.lvd1689m",
        "vit_large_patch16_dinov3",
    )),
    "retfound": dict(embed_dim=1024, patch_size=16, timm_names=(
        "vit_large_patch16_224.mae",
        "vit_large_patch16_224",
    )),
}


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable rank-``r`` update.

    ``y = W x + (alpha / r) B A x`` with ``B`` zero-initialised, so the wrapped layer
    starts as an exact copy of the one it replaces and only the adapters train.

    Args:
        base: The linear layer to wrap. Its weights are frozen in place.
        rank: Rank of the update.
        alpha: Scaling numerator; the update is scaled by ``alpha / rank``.
    """

    def __init__(self, base, rank=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = alpha / rank

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(x))


def inject_lora(vit, rank=8, alpha=16.0):
    """Wrap every attention projection of a ViT in a :class:`LoRALinear`.

    Freezes everything else, so after this call the only trainable tensors in ``vit``
    are the adapters.

    Args:
        vit: A ViT whose blocks expose ``attn.qkv`` and ``attn.proj`` linear layers.
        rank: Rank of each adapter.
        alpha: Scaling numerator of each adapter.

    Returns:
        int: Number of layers wrapped, so a caller can fail loudly on zero rather than
        train nothing at all.
    """
    for p in vit.parameters():
        p.requires_grad = False
    wrapped = 0
    for module in vit.modules():
        attn = getattr(module, "attn", None)
        if attn is None:
            continue
        for name in ("qkv", "proj"):
            layer = getattr(attn, name, None)
            if isinstance(layer, nn.Linear):
                setattr(attn, name, LoRALinear(layer, rank=rank, alpha=alpha))
                wrapped += 1
    return wrapped


def load_vit_weights(weights, map_location="cpu"):
    """Resolve a backbone-weights argument to something :func:`_build_vit` can load.

    Accepts a state dict or a path to a ``.pth`` / ``.pt`` / ``.bin`` (torch) or a
    ``.safetensors`` file. Both formats are live: DINOv3 is staged as safetensors, which
    is what ``hf download`` takes from a hub repo that offers both, and RETFound ships a
    torch ``.pth``. See the backbone-study section of the README for the two downloads.

    Args:
        weights: State dict, path, or ``None``.
        map_location: Device the torch checkpoint is read onto.

    Returns:
        The state dict, or the input unchanged when it is already one.
    """
    if weights is None or not isinstance(weights, (str, os.PathLike)):
        return weights
    path = os.fspath(weights)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location=map_location, weights_only=False)


def frozen_backbone_dtype(name, amp_dtype):
    """Pick the dtype a *frozen* backbone's weights are cast to.

    A frozen ViT is cast outright instead of running under autocast, which gives up the
    one thing autocast guarantees: layer norm and softmax stay in fp32. DINOv3 does not
    survive that. Its blocks carry LayerScale (``gamma_1`` / ``gamma_2``) and it has no
    bounded position embedding to keep activations small, and in fp16 the whole feature
    map comes back NaN -- measured at 560 px on sm_89, every pixel, against a DINOv2
    control that is finite in fp16 on the same run. bf16 has fp32's exponent range and
    lands within 1% of fp32 end to end, so it is substituted for fp16 here.

    Only the frozen path needs this. ``unfrozen`` and ``lora`` keep their weights in fp32
    and their autocast forward is finite in fp16, gradients included.

    Args:
        name: A key of :data:`VIT_BACKBONES`.
        amp_dtype: The dtype the caller asked for.

    Returns:
        torch.dtype: ``amp_dtype`` unchanged, or ``torch.bfloat16`` in place of
        ``torch.float16`` for a backbone that overflows it.
    """
    if amp_dtype is not torch.float16 or not VIT_BACKBONES[name].get("fp16_overflows"):
        return amp_dtype
    print(f"[ViTPatchBackbone] frozen {name}: substituting bfloat16 for the requested "
          "float16, which overflows to NaN in this backbone (needs sm_80 or newer to be "
          "fast; it is correct but emulated below that)")
    return torch.bfloat16


def _build_vit(name, weights):
    """Instantiate one backbone and load its pretrained weights.

    Args:
        name: A key of :data:`VIT_BACKBONES`.
        weights: A state dict or a path to one, already resolved by
            :func:`load_vit_weights`. ``None`` leaves the model at its own
            initialisation, which is only useful for a smoke test.

    Returns:
        nn.Module: The ViT, in eval mode, with a ``forward_features`` that returns dense
        patch tokens (the two families are unified in :meth:`ViTPatchBackbone.tokens`).
    """
    if name == "dinov2":
        vit = vit_large(img_size=518, patch_size=14, init_values=1.0,
                        ffn_layer="mlp", block_chunks=0).eval()
        if weights is not None:
            vit.load_state_dict(weights)
        return vit

    names = VIT_BACKBONES[name]["timm_names"]
    available = [n for n in names if n.split(".")[0] in set(timm.list_models())]
    if not available:
        raise ValueError(
            f"none of timm's names for {name!r} ({', '.join(names)}) exist in timm "
            f"{timm.__version__}; upgrade timm or add the current name to "
            "VIT_BACKBONES"
        )
    vit = timm.create_model(available[0], pretrained=False, num_classes=0,
                            dynamic_img_size=True).eval()
    if weights is not None:
        for key in ("model", "state_dict", "teacher"):
            if isinstance(weights, dict) and key in weights:
                weights = weights[key]
                break
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
        missing, unexpected = vit.load_state_dict(weights, strict=False)
        blocking = [k for k in missing if not k.startswith("head")]
        if blocking:
            raise ValueError(
                f"backbone {name!r}: {len(blocking)} weights missing from the checkpoint, "
                f"first few {blocking[:5]}; this is the wrong file for this architecture"
            )
        if unexpected:
            print(f"[encoder] {name}: ignored {len(unexpected)} unexpected checkpoint keys")
    return vit


class ViTPatchBackbone(nn.Module):
    """The coarse branch: one ViT, exposed as a dense ``(B, C, H/p, W/p)`` feature map.

    Args:
        name: A key of :data:`VIT_BACKBONES`.
        weights: State dict (or path to one) for that backbone.
        tuning: ``frozen`` / ``unfrozen`` / ``lora``; see the module docstring.
        amp_dtype: Dtype the frozen backbone runs in, subject to
            :func:`frozen_backbone_dtype`. Ignored when it is trainable, which stays in
            fp32 and relies on autocast instead -- a trainable fp16 weight has no master
            copy to accumulate small updates into.
        lora_rank, lora_alpha: Adapter shape, read only by ``tuning='lora'``.
    """

    def __init__(self, name="dinov2", weights=None, tuning="frozen",
                 amp_dtype=torch.float16, lora_rank=8, lora_alpha=16.0):
        super().__init__()
        if name not in VIT_BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; expected one of "
                             f"{', '.join(VIT_BACKBONES)}")
        if tuning not in ("frozen", "unfrozen", "lora"):
            raise ValueError(f"unknown tuning {tuning!r}; expected frozen / unfrozen / lora")
        spec = VIT_BACKBONES[name]
        self.name = name
        self.tuning = tuning
        self.embed_dim = spec["embed_dim"]
        self.patch_size = spec["patch_size"]
        self.amp_dtype = (frozen_backbone_dtype(name, amp_dtype) if tuning == "frozen"
                          else amp_dtype)

        vit = _build_vit(name, weights)
        vit.requires_grad_(tuning != "frozen")
        if tuning == "lora":
            wrapped = inject_lora(vit, rank=lora_rank, alpha=lora_alpha)
            if wrapped == 0:
                raise ValueError(
                    f"backbone {name!r}: found no attn.qkv/attn.proj to adapt, so "
                    "tuning='lora' would train nothing"
                )
        if tuning == "frozen":
            self._frozen_vit = [vit.to(self.amp_dtype)]
            self.vit = None
        else:
            self._frozen_vit = []
            self.vit = vit

    def _vit(self, device):
        """The ViT, moved to ``device`` on first use (the frozen one is not a submodule,
        so ``.to()`` on the parent does not reach it)."""
        if self.vit is not None:
            return self.vit
        vit = self._frozen_vit[0]
        if next(vit.parameters()).device != device:
            vit = vit.to(device).to(self.amp_dtype)
            self._frozen_vit[0] = vit
        return vit

    def tokens(self, vit, x, grid):
        """Dense patch tokens as ``(B, C, h, w)``, over both backbone families.

        Args:
            vit: The instantiated backbone.
            x: Input images, ``(B, 3, H, W)``.
            grid: ``(H // patch, W // patch)``.

        Returns:
            torch.Tensor: ``(B, embed_dim, h, w)``.
        """
        h, w = grid
        if self.name == "dinov2":
            patch = vit.forward_features(x)["x_norm_patchtokens"]
        else:
            patch = vit.forward_features(x)[:, vit.num_prefix_tokens:]
        return patch.permute(0, 2, 1).reshape(x.shape[0], self.embed_dim, h, w)

    def train(self, mode=True):
        super().train(mode)
        for vit in self._frozen_vit:
            vit.eval()
        return self

    def forward(self, x):
        B, C, H, W = x.shape
        grid = (H // self.patch_size, W // self.patch_size)
        vit = self._vit(x.device)
        if self.tuning == "frozen":
            with torch.no_grad():
                return self.tokens(vit, x.to(self.amp_dtype), grid)
        return self.tokens(vit, x, grid)


class VGG19(nn.Module):
    def __init__(self, pretrained=True, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if pretrained:
            weights = tvm.vgg.VGG19_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.layers = nn.ModuleList(tvm.vgg19_bn(weights=weights).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandViT(nn.Module):
    """VGG19 pyramid (scales 1-8) plus a ViT for the coarse scale-16 branch.

    Args:
        cnn_kwargs: Forwarded to :class:`VGG19`.
        amp: Whether the CNN branch runs under autocast.
        vit_weights: State dict (or path) for the chosen backbone. Named generically
            because it is no longer necessarily DINOv2's.
        amp_dtype: Dtype a frozen backbone runs in.
        backbone: A key of :data:`VIT_BACKBONES`.
        backbone_tuning: ``frozen`` / ``unfrozen`` / ``lora``.
        lora_rank, lora_alpha: Adapter shape for ``backbone_tuning='lora'``.
    """

    def __init__(self, cnn_kwargs=None, amp=False, vit_weights=None,
                 amp_dtype=torch.float16, backbone="dinov2", backbone_tuning="frozen",
                 lora_rank=8, lora_alpha=16.0):
        super().__init__()
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.backbone = ViTPatchBackbone(
            name=backbone, weights=vit_weights, tuning=backbone_tuning,
            amp_dtype=amp_dtype if amp else torch.float32,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )

    @property
    def patch_size(self):
        return self.backbone.patch_size

    @property
    def coarse_dim(self):
        """Channel width of the scale-16 branch, i.e. what ``proj16`` must accept."""
        return self.backbone.embed_dim

    def train(self, mode: bool = True):
        self.cnn.train(mode)
        self.backbone.train(mode)
        return self

    def forward(self, x, upsample = False):
        feature_pyramid = self.cnn(x)
        if not upsample:
            feature_pyramid[16] = self.backbone(x)
        return feature_pyramid


class CNNandDinov2(CNNandViT):
    """``CNNandViT`` pinned to RoMa's own frozen DINOv2, for the vanilla entry points.

    Kept so ``roma_outdoor`` / ``roma_indoor`` / ``roma_model_pad`` read as before; the
    backbone study goes through ``CNNandViT`` instead.
    """

    def __init__(self, cnn_kwargs=None, amp=False, dinov2_weights=None,
                 amp_dtype=torch.float16):
        super().__init__(cnn_kwargs=cnn_kwargs, amp=amp, vit_weights=dinov2_weights,
                         amp_dtype=amp_dtype, backbone="dinov2", backbone_tuning="frozen")
