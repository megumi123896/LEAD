"""Train the modified Latent-LEAD Swin-B model on LAU-GRES.

By default, the dataset is discovered next to the project directory and
outputs are written under results/lau_gres_8cls.
"""

import os
import random
from pathlib import Path

# Dataset registration happens during imports below, so set roots first.
_dataset_root = Path(__file__).resolve().parent.parent / "LAU-GRES"
os.environ.setdefault("LAU_GRES_ROOT", str(_dataset_root / "LAU-GRES"))
os.environ.setdefault("LAU_GRES_IMAGE_ROOT", str(_dataset_root / "images"))

import numpy as np
import torch
import torch.nn.functional as F

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from gres_model import LEAD, add_maskformer2_config, add_refcoco_config
from gres_model.data.dataset_mappers.refcoco_mapper_lau_gres import LAUGRESMapper
from gres_model.data.datasets import register_lau_gres  # noqa: F401
from gres_model.evaluation.refer_evaluation_lau_gres import LAUGRESEvaluator
import gres_model.modeling.criterion as criterion_module
from gres_model.modeling.transformer_decoder.query_generation import (
    Query_Embedding_Padding_Simple,
)
from train_net import Trainer as OriginalTrainer


DEFAULT_TRAIN_CONFIG = "configs/referring_swin_base_latent_lau_gres_train.yaml"


def prepare_targets_lau_gres(self, batched_inputs, images):
    """Build eight-class count and segmentation targets, including no-target data."""
    h_pad, w_pad = images.tensor.shape[-2:]
    new_targets = []

    for sample in batched_inputs:
        labels = torch.zeros(self.num_classes, device=self.device)
        for category_id in sample["category_id"]:
            if category_id != self.num_classes:
                labels[category_id] += 1

        instances = sample["instances"].to(self.device)
        gt_masks = instances.gt_masks.tensor
        padded_masks = torch.zeros(
            (gt_masks.shape[0], h_pad, w_pad),
            dtype=gt_masks.dtype,
            device=gt_masks.device,
        )
        if gt_masks.shape[0] > 0:
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

        is_empty = torch.tensor(
            sample["empty"],
            dtype=instances.gt_classes.dtype,
            device=instances.gt_classes.device,
        )
        target = {
            "labels": labels,
            "masks": padded_masks,
            "empty": is_empty,
        }
        if sample["gt_mask_merged"] is not None:
            target["gt_mask_merged"] = sample["gt_mask_merged"].to(self.device)
        new_targets.append(target)

    return new_targets


def padding_without_inplace(self, lang_feat, lang_sent, lang_mask):
    """Replace PAD positions with sentence semantics without an in-place write."""
    _, token_count, _ = lang_feat.shape
    lengths = lang_mask.squeeze(-1).sum(dim=1).long()
    positions = torch.arange(token_count, device=lang_feat.device).unsqueeze(0)
    padding_positions = positions >= lengths.unsqueeze(1)
    return torch.where(
        padding_positions.unsqueeze(-1),
        lang_sent.unsqueeze(1),
        lang_feat,
    )


def counting_loss_lau_gres(inputs, targets):
    """Use the previous LAU-GRES eight-class Smooth-L1 count loss."""
    loss = F.smooth_l1_loss(inputs, targets, reduction="none")
    weights = torch.ones(
        inputs.shape[-1],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    return torch.mean(loss * weights)


# These adaptations are intentionally local to the LAU-GRES entry points.
LEAD.prepare_targets = prepare_targets_lau_gres
Query_Embedding_Padding_Simple.padding = padding_without_inplace
criterion_module.count_loss_jit = counting_loss_lau_gres


class LAUGRESLatentTrainer(OriginalTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=LAUGRESMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=LAUGRESMapper(cfg, False)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
        return LAUGRESEvaluator(
            dataset_name,
            distributed=True,
            output_dir=output_folder,
        )


def setup_config(args, default_config=DEFAULT_TRAIN_CONFIG, logger_name="lau_gres_train"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file(args.config_file or default_config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name=logger_name,
    )
    return cfg


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    cfg = setup_config(args)
    seed_everything(cfg.SEED)

    if args.eval_only:
        raise ValueError(
            "This is the training entry point; use eval_lau_gres_latent.py for evaluation."
        )

    trainer = LAUGRESLatentTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
