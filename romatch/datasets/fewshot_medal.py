"""
Episodic wrapper around :class:`~romatch.datasets.medal.MedalPairDataset` for
few-shot registration training entirely from self-supervised synthetic
MeDAL-Retina pairs.

Mirrors :class:`~romatch.datasets.fewshot_homography_octa.FewShotHomographyOctaDataset`:
each ``__getitem__`` draws one **query** pair (registered + supervised) and one
**support** pair (its dense GT warp is the modality prior), all from the same pool --
``Auxilliary_Training`` for ``split="train"``, ``SuperRetina_training_data`` for
``split in ("val", "test")``. Query and support pairs all carry a dense warp,
analytically generated rather than TPS-fitted from annotations.
"""

import random

import torch

from romatch.datasets.medal import MedalPairDataset


class FewShotMedalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root="data",
        subset="MeDAL-Retina",
        split="train",
        ht=560,
        wt=560,
        samples_per_epoch=None,
        **base_kwargs,
    ):
        super().__init__()
        self.subset = subset
        self.dataset = MedalPairDataset(
            data_root=data_root, subset=subset, split=split, ht=ht, wt=wt, **base_kwargs,
        )
        # One query plus one distinct support have to fit in the pool.
        if len(self.dataset) < 2:
            raise ValueError(
                f"Few-shot MeDAL dataset needs at least 2 images for one query + one "
                f"support, found {len(self.dataset)}"
            )
        self._length = samples_per_epoch or len(self.dataset)
        print(
            f"[FewShotMedal] split={split}, {len(self.dataset)} images, "
            f"1-shot episodes, epoch length {self._length}"
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # idx is ignored: each item is a fresh random episode (train), or a
        # reproducible one keyed by the drawn indices (val/test, since
        # MedalPairDataset seeds its transform from the sample index).
        ds = self.dataset
        q_idx, sup_idx = random.sample(range(len(ds)), 2)
        q, sup = ds[q_idx], ds[sup_idx]

        return {
            "im_A": q["im_A"],
            "im_B": q["im_B"],
            "dense_warp_A_to_B": q["dense_warp_A_to_B"],
            "supp_im_A": sup["im_A"],
            "supp_im_B": sup["im_B"],
            "supp_dense_warp_A_to_B": sup["dense_warp_A_to_B"],
            "subset": self.subset,
        }


def build_fewshot_medal(
    data_root="data",
    subset="MeDAL-Retina",
    split="train",
    ht=560,
    wt=560,
    **kwargs,
):
    """Convenience builder; wrap the returned dataset in a standard DataLoader."""
    return FewShotMedalDataset(
        data_root=data_root, subset=subset, split=split, ht=ht, wt=wt, **kwargs,
    )
