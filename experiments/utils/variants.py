"""Canonical configuration for the RoMa ablation arms.

Train and test entry points use this table to build the model and label its result.
Every arm is 1-shot: one support pair conditions each episode, at train time and at
test time alike.

===========================  ========  ============  ============  =========
variant                      finetune  use_support   use_perturb   use_freq
===========================  ========  ============  ============  =========
``zero``                     no        -             -             -
``baseline``                 yes       -             -             -
``few_shot``                 yes       yes           -             -
``few_shot+perturb``         yes       yes           yes           -
``few_shot+freq``            yes       yes           -             yes
``few_shot+perturb+freq``    yes       yes           yes           yes
``gamma0+perturb``           yes       -             yes           -
``gamma0+freq``              yes       -             -             yes
``gamma0+perturb+freq``      yes       -             yes           yes
===========================  ========  ============  ============  =========

``zero`` uses official weights without fine-tuning. ``baseline`` fine-tunes the same
vanilla graph. ``gamma0`` arms omit the support residual while retaining the named
perturbation/filter components. Results are stored under ``metrics`` and identified by
``_meta.variant``.
"""

# finetune:    load official RoMa weights and train, or score them as they are.
# use_support: build the ridge residual (phi/psi/gamma) and encode the support pair.
# use_perturb: training-only channel-stat perturbation of the im_B features.
# use_freq:    shared low/mid/high frequency modulation of every feature branch.
VARIANT_SPECS = {
    "zero":                  dict(finetune=False, use_support=False, use_perturb=False, use_freq=False),
    "baseline":              dict(finetune=True,  use_support=False, use_perturb=False, use_freq=False),
    "few_shot":              dict(finetune=True,  use_support=True,  use_perturb=False, use_freq=False),
    "few_shot+perturb":      dict(finetune=True,  use_support=True,  use_perturb=True,  use_freq=False),
    "few_shot+freq":         dict(finetune=True,  use_support=True,  use_perturb=False, use_freq=True),
    "few_shot+perturb+freq": dict(finetune=True,  use_support=True,  use_perturb=True,  use_freq=True),
    "gamma0+perturb":        dict(finetune=True,  use_support=False, use_perturb=True,  use_freq=False),
    "gamma0+freq":           dict(finetune=True,  use_support=False, use_perturb=False, use_freq=True),
    "gamma0+perturb+freq":   dict(finetune=True,  use_support=False, use_perturb=True,  use_freq=True),
}

VARIANTS = tuple(VARIANT_SPECS)

# The arm every other one is measured against: same fine-tuning, no adaptation.
REFERENCE_VARIANT = "baseline"

# The default keeps the pre-ablation behaviour of these entry points, so an existing
# command line that names no variant still builds the model it used to.
DEFAULT_VARIANT = "few_shot+perturb+freq"


def variant_spec(variant):
    """Look up one arm's switches.

    Args:
        variant: An arm name from :data:`VARIANTS`.

    Returns:
        dict: The arm's ``finetune`` / ``use_support`` / ``use_perturb`` / ``use_freq``
        switches.

    Raises:
        ValueError: If the name is not one of the nine arms.
    """
    try:
        return dict(VARIANT_SPECS[variant])
    except KeyError:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {', '.join(VARIANTS)}"
        ) from None


def align_scales_for(variant, scales):
    """Resolve the scales the alignment modules are inserted at.

    Args:
        variant: An arm name from :data:`VARIANTS`.
        scales: Scales asked for on the command line.

    Returns:
        tuple[int, ...]: The requested scales, or an empty tuple for the arms that
        build no alignment module.
    """
    spec = variant_spec(variant)
    enabled = spec["use_support"] or spec["use_perturb"] or spec["use_freq"]
    return tuple(int(s) for s in scales) if enabled else ()


def align_kwargs_for(variant, gamma_init=0.0, perturb_noise_std=0.0):
    """Build the ``align_kwargs`` that construct one arm's alignment modules.

    Args:
        variant: An arm name from :data:`VARIANTS`.
        gamma_init: Residual-gate init; read only by the support-conditioned arms.
        perturb_noise_std: Perturbation magnitude; read only when ``use_perturb`` is on.

    Returns:
        dict: Keyword arguments for ``SupportAlignmentModule`` /
        ``FewShotRegressionMatcher``.
    """
    spec = variant_spec(variant)
    return {
        "use_support": spec["use_support"],
        "use_perturb": spec["use_perturb"],
        "use_freq": spec["use_freq"],
        "gamma_init": gamma_init,
        "perturb_noise_std": perturb_noise_std,
    }


def uses_support(variant):
    """Whether the arm conditions on a support prior.

    Args:
        variant: An arm name from :data:`VARIANTS`.

    Returns:
        bool: True for the ``few_shot*`` arms.
    """
    return variant_spec(variant)["use_support"]


def is_finetuned(variant):
    """Whether the arm is trained, or scored straight off the official weights.

    Args:
        variant: An arm name from :data:`VARIANTS`.

    Returns:
        bool: False only for ``zero``.
    """
    return variant_spec(variant)["finetune"]


def add_variant_arg(parser, default=DEFAULT_VARIANT):
    """Register ``--variant`` on a train or test entry point.

    Args:
        parser: The argument parser to extend.
        default: The arm to build when none is named.
    """
    parser.add_argument(
        "--variant", default=default, choices=list(VARIANTS),
        help="ablation arm; decides the model that is built and the label written into "
             "the result JSON. Train and test must name the same one -- an augmentation "
             "and a module that co-adapts with the decoder cannot be switched on after "
             "the fact. See experiments/utils/variants.py for the table.",
    )


def describe(variant):
    """One-line summary of an arm, for a run header.

    Args:
        variant: An arm name from :data:`VARIANTS`.

    Returns:
        str: The arm name followed by its switches.
    """
    spec = variant_spec(variant)
    on = [name for name in ("use_support", "use_perturb", "use_freq") if spec[name]]
    return (f"{variant} (finetune={spec['finetune']}, "
            f"{'+'.join(on) if on else 'no adaptation'})")
