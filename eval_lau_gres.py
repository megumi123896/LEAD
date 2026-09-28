"""Evaluate a LEAD checkpoint on LAU-GRES.

Default checkpoint: results/lau_gres_8cls/model_final.pth
Default output: results/lau_gres_8cls/eval_model_final
"""

import os
from pathlib import Path

_dataset_root = Path(__file__).resolve().parent.parent / "LAU-GRES"
os.environ.setdefault("LAU_GRES_ROOT", str(_dataset_root / "LAU-GRES"))
os.environ.setdefault("LAU_GRES_IMAGE_ROOT", str(_dataset_root / "images"))

import torch

import detectron2.utils.comm as comm           
from detectron2.checkpoint import DetectionCheckpointer    
from detectron2.config import get_cfg   
from detectron2.data import build_detection_test_loader   
from detectron2.engine import default_argument_parser, default_setup, launch   
from detectron2.projects.deeplab import add_deeplab_config  
from detectron2.utils.logger import setup_logger    

from gres_model import add_maskformer2_config, add_refcoco_config     
from gres_model.data.dataset_mappers.refcoco_mapper_lau_gres import LAUGRESMapper    
from gres_model.data.datasets import register_lau_gres    
from gres_model.evaluation.refer_evaluation_lau_gres import LAUGRESEvaluator    
from gres_model.modeling.transformer_decoder.query_generation import (       
    Query_Embedding_Padding_Simple,
)
from train_net import Trainer as OriginalTrainer     


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


class LAUGRESEvalTrainer(OriginalTrainer):
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
    cfg.merge_from_file("configs/referring_swin_base_eval.yaml")

    # TRAIN is kept non-empty because LEAD reads TRAIN[0] for metadata even
    # during eval-only runs. It will not train when this script runs.
    cfg.DATASETS.TRAIN = ("lau_gres_train",)
    cfg.DATASETS.TEST = ("lau_gres_test",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
    cfg.MODEL.WEIGHTS = "results/lau_gres_8cls/model_final.pth"
    cfg.REFERRING.BERT_TYPE = "pretrain/bert"
    cfg.OUTPUT_DIR = "results/lau_gres_8cls/eval_model_final"
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="lau_gres_eval")
    return cfg


def main(args):
    cfg = setup(args)
    if not args.eval_only:
        raise ValueError("eval_lau_gres.py is evaluation-only; please add --eval-only")

    model = LAUGRESEvalTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    return LAUGRESEvalTrainer.test(cfg, model)


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
