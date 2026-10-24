"""
Episodic wrapper around :class:`HomographyOctaDataset` for few-shot cross-domain
registration.

Each ``__getitem__`` draws two pairs **from the same subset (modality)**: one **query**
pair (registered + supervised) and one **support** pair (its GT warp is the modality
prior). The query pair keeps RoMa's standard batch keys (``im_A`` / ``im_B`` /
``dense_warp_A_to_B``) so the training loop and ``RobustLosses`` are unchanged; the
support pair is emitted under ``supp_*`` keys and always carries a dense warp
(``supp_dense_warp_A_to_B``) and a control-point neighborhood mask
(``supp_ctrl_mask``) consumed by ``SupportAlignmentModule``.

Cross-domain (FS-SAM2 style): train on one benchmark and test few-shot on unseen ones --
e.g. train on **FIRE** and evaluate **OCTA / MISP / FA-SLO** in separate runs. Every
dataset instance represents exactly one domain, so support and query always share it.
"""

import random

import torch

from romatch.datasets.homography_octa import HomographyOctaDataset

class FewShotHomographyOctaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root="data",
        subset="FIRE",
        fold=0,
        split="train",
        ht=560,
        wt=560,
        samples_per_epoch=None,
        **base_kwargs,
    ):
        super().__init__()
        self.ht, self.wt = ht, wt
        self.subset = subset
        self.dataset = HomographyOctaDataset(
            data_root=data_root, subsets=(self.subset,), fold=fold, split=split,
            ht=ht, wt=wt, **base_kwargs,
        )
        # One query plus one distinct support have to fit in the subset.
        if len(self.dataset) < 2:
            raise ValueError(
                f"Few-shot dataset {self.subset!r} needs at least 2 pairs for one "
                f"query + one support, found {len(self.dataset)}"
            )
        self._length = samples_per_epoch or len(self.dataset)
        print(
            f"[FewShot] subset {self.subset}, {len(self.dataset)} pairs, "
            f"1-shot episodes, epoch length {self._length}"
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # idx is ignored: each item is a fresh random episode. Distinct indices ensure
        # the support excludes the query, and both come from the one configured dataset.
        s = self.subset
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
            "supp_ctrl_mask": sup["ctrl_mask"],
            "subset": s,
        }


def build_fewshot_homography_octa(
    data_root="data",
    subset="OCTA",
    fold=0,
    split="train",
    ht=560,
    wt=560,
    **kwargs,
):
    """Convenience builder; wrap the returned dataset in a standard DataLoader."""
    return FewShotHomographyOctaDataset(
        data_root=data_root, subset=subset, fold=fold, split=split,
        ht=ht, wt=wt, **kwargs,
    )
