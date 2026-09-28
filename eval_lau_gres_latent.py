"""Evaluate a modified Latent-LEAD checkpoint on the LAU-GRES test split."""

import os
from pathlib import Path

# Set roots before importing the shared trainer, which registers the dataset.
_dataset_root = Path(__file__).resolve().parent.parent / "LAU-GRES"
os.environ.setdefault("LAU_GRES_ROOT", str(_dataset_root / "LAU-GRES"))
os.environ.setdefault("LAU_GRES_IMAGE_ROOT", str(_dataset_root / "images"))

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import verify_results

from train_lau_gres_latent import (
    LAUGRESLatentTrainer,
    seed_everything,
    setup_config,
)


DEFAULT_EVAL_CONFIG = "configs/referring_swin_base_latent_lau_gres_eval.yaml"


def main(args):
    if not args.eval_only:
        raise ValueError("Evaluation requires the --eval-only argument.")

    cfg = setup_config(
        args,
        default_config=DEFAULT_EVAL_CONFIG,
        logger_name="lau_gres_eval",
    )
    seed_everything(cfg.SEED)

    model = LAUGRESLatentTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=args.resume,
    )
    results = LAUGRESLatentTrainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, results)
    return results


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
