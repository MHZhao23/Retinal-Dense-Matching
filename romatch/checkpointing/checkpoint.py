import gc
import json
import math
import os

import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger

import romatch


class CheckPoint:
    """Save and restore a training run.

    A run is resumable only if what it saves is enough to continue from. The model
    weights alone are not: the optimizer moments, the LR schedule, the AMP loss scale
    and the data sampler's RNG would all be rebuilt from nothing, which continues along
    a different trajectory from the one that was interrupted -- a sampler left at its
    seed replays the batch order from step 0 rather than carrying on, and an LR
    schedule counting from zero never reaches a milestone set in global steps. So a
    save carries all of them, plus the best validation score seen so far -- without
    that, a resumed run overwrites a good ``_best.pth`` with the first worse score it
    happens to measure.

    Each save also writes ``<name>_progress.json`` beside the checkpoint, recording the
    step reached and the step count aimed at. It exists for the sweep scripts, which
    read it with ``sed`` and skip an arm that is already finished without paying for a
    model build to find that out.

    Both files are written to a temporary path and renamed into place. These runs die
    mid-flight -- that is what resuming is for -- and a process killed inside
    ``torch.save`` would otherwise leave a truncated checkpoint that costs the next run
    its entire history.
    """

    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self.progress_path = self._path("_progress.json")
        self._progress = self._read_progress()

    def _path(self, suffix):
        """Absolute path of one of this checkpoint's files."""
        return os.path.join(self.dir, self.name + suffix)

    def _read_progress(self):
        """The progress file's contents, or an empty dict when it is absent or bad."""
        try:
            with open(self.progress_path, encoding="utf-8") as file:
                loaded = json.load(file)
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _write_progress(self, **updates):
        """Merge ``updates`` into the progress file and rewrite it atomically."""
        self._progress.update(updates)
        tmp = self.progress_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(self._progress, file, indent=2)
        os.replace(tmp, self.progress_path)

    @staticmethod
    def _atomic_save(states, path):
        """Write a checkpoint so that an interrupted save cannot destroy the old one."""
        tmp = path + ".tmp"
        torch.save(states, tmp)
        os.replace(tmp, path)

    @staticmethod
    def _unwrap(model):
        """The underlying module of a possibly data-parallel wrapper."""
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            return model.module
        return model


    def record_progress(self, step, total_steps=None):
        """Stamp the progress file without writing a checkpoint.

        A run that resumes only to find nothing left to do never reaches ``save``, so
        without this an arm that finished before the progress file existed would
        re-enter Python on every sweep just to discover its step range is empty.
        """
        if romatch.RANK == 0:
            fields = {"step": int(step)}
            if total_steps is not None:
                fields["total_steps"] = int(total_steps)
            self._write_progress(**fields)
    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        grad_scaler=None,
        data_generator=None,
        best_metric=None,
        total_steps=None,
        ):
        if romatch.RANK == 0:
            assert model is not None
            model = self._unwrap(model)
            states = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "n": n,
            }
            if grad_scaler is not None:
                states["grad_scaler"] = grad_scaler.state_dict()
            if data_generator is not None:
                # Taken after the block's loader is exhausted, so it is the state the
                # next block would have started from had nothing interrupted the run.
                states["data_generator"] = data_generator.get_state()
            if best_metric is not None:
                states["best_metric"] = float(best_metric)
            self._atomic_save(states, self._path("_latest.pth"))

            progress = {"step": int(n)}
            if total_steps is not None:
                progress["total_steps"] = int(total_steps)
            if best_metric is not None and math.isfinite(best_metric):
                progress["best_metric"] = float(best_metric)
            self._write_progress(**progress)
            logger.info(f"Saved states {list(states.keys())}, at step {n}")

    def save_best(self, model, n, metric=None):
        if romatch.RANK == 0:
            model = self._unwrap(model)
            states = {"model": model.state_dict(), "n": n}
            if metric is not None:
                states["metric"] = float(metric)
            self._atomic_save(states, self._path("_best.pth"))
            updates = {"best_step": int(n)}
            if metric is not None and math.isfinite(metric):
                updates["best_metric"] = float(metric)
            self._write_progress(**updates)
            logger.info(f"Saved best model at step {n}")

    def load(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        grad_scaler=None,
        data_generator=None,
        best_metric=float("inf"),
        ):
        path = self._path("_latest.pth")
        if os.path.exists(path) and romatch.RANK == 0:
            try:
                states = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.error(
                    f"Could not read {path} ({e}); starting from step 0. "
                    "Delete the file to silence this."
                )
                return model, optimizer, lr_scheduler, n, best_metric
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            if grad_scaler is not None and "grad_scaler" in states:
                grad_scaler.load_state_dict(states["grad_scaler"])
            if data_generator is not None and "data_generator" in states:
                try:
                    data_generator.set_state(states["data_generator"])
                except Exception as e:
                    # A state saved by a different torch build can be the wrong size.
                    # The seeded generator it falls back to still samples the same
                    # distribution, so warn and carry on rather than fail the run.
                    print(f"Failed to load states for data_generator, with error {e}")
            if "best_metric" in states:
                best_metric = states["best_metric"]
            print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n, best_metric
