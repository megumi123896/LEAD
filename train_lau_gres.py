"""Train LEAD-Swin-B on the LAU-GRES GRefCOCO-style dataset.

By default, the dataset is discovered next to the project directory and
outputs are written under results/lau_gres_8cls.
"""

import os
import random
from pathlib import Path

_dataset_root = Path(__file__).resolve().parent.parent / "LAU-GRES"
os.environ.setdefault("LAU_GRES_ROOT", str(_dataset_root / "LAU-GRES"))
os.environ.setdefault("LAU_GRES_IMAGE_ROOT", str(_dataset_root / "images"))

import numpy as np
import torch
import torch.nn.functional as F

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
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


def prepare_targets_lau_gres(self, batched_inputs, images):
    """Original prepare_targets adapted for BitMasks and no-target samples."""
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


LEAD.prepare_targets = prepare_targets_lau_gres


def padding_without_inplace(self, lang_feat, lang_sent, lang_mask):
    """PyTorch 2.x-safe replacement for the original in-place token padding."""
    _, token_count, _ = lang_feat.shape
    lengths = lang_mask.squeeze(-1).sum(dim=1).long()
    positions = torch.arange(token_count, device=lang_feat.device).unsqueeze(0)
    padding_positions = positions >= lengths.unsqueeze(1)
    return torch.where(
        padding_positions.unsqueeze(-1),
        lang_sent.unsqueeze(1),
        lang_feat,
    )


Query_Embedding_Padding_Simple.padding = padding_without_inplace


def counting_loss_lau_gres(inputs, targets):
    """LAU-GRES-compatible count loss.

    The original LEAD loss hard-codes 12 class weights. LAU-GRES uses
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8, so the count loss must follow the
    current count vector width.
    """
    loss = F.smooth_l1_loss(inputs, targets, reduction="none")
    weights = torch.ones(
        inputs.shape[-1],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    return torch.mean(loss * weights)


# Keep the original criterion.py untouched; only this LAU-GRES entry point uses
# the dynamic-width count loss.
criterion_module.count_loss_jit = counting_loss_lau_gres


class LAUGRESTrainer(OriginalTrainer):
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
        return LAUGRESEvaluator(dataset_name, distributed=True, output_dir=output_folder)


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_refcoco_config(cfg)
    cfg.merge_from_file("configs/referring_swin_base.yaml")

    cfg.DATASETS.TRAIN = ("lau_gres_train",)
    cfg.DATASETS.TEST = ("lau_gres_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
    cfg.MODEL.WEIGHTS = "pretrain/swin_base_patch4_window12_384_22k.pkl"
    cfg.REFERRING.BERT_TYPE = "pretrain/bert"
    cfg.OUTPUT_DIR = "results/lau_gres_8cls"
    cfg.DATALOADER.NUM_WORKERS = 0

    # Original GRefCOCO-Swin-B schedule:
    # 78930 samples, global batch 48, 120000 iters ~= 72.98 epochs.
    # LAU-GRES has 14254 training sentence samples. Batch 1 is the safe
    # default for Swin-B on a 12GB GPU.
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 5.0e-5 / 48.0
    cfg.SOLVER.COS_END_LR = 5.0e-7 / 48.0
    cfg.SOLVER.MAX_ITER = 1040201
    cfg.SOLVER.WARMUP_ITERS = 59
    # 8 epochs * 14254 samples / batch 1.
    cfg.SOLVER.CHECKPOINT_PERIOD = 114032
    cfg.TEST.EVAL_PERIOD = 114032

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="lau_gres")
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    cfg = setup(args)
    seed_everything(cfg.SEED)

    if args.eval_only:
        model = LAUGRESTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return LAUGRESTrainer.test(cfg, model)

    trainer = LAUGRESTrainer(cfg)
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
