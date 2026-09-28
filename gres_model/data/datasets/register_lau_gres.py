"""Register the LAU-GRES GRefCOCO-style dataset without touching originals."""

import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

from .lau_gres_loader import load_lau_gres_json


def register_lau_gres(data_root, image_root):
    for split in ["train", "val", "test"]:
        dataset_id = f"lau_gres_{split}"
        if dataset_id in DatasetCatalog:
            continue
        DatasetCatalog.register(
            dataset_id,
            lambda split=split, data_root=data_root, image_root=image_root: load_lau_gres_json(
                data_root, split, image_root
            ),
        )
        MetadataCatalog.get(dataset_id).set(
            evaluator_type="refer",
            dataset_name="lau_gres",
            split=split,
            root=data_root,
            image_root=image_root,
        )


_project_root = Path(__file__).resolve().parents[3]
_dataset_root = _project_root.parent / "LAU-GRES"
_data_root = os.environ.get(
    "LAU_GRES_ROOT", str(_dataset_root / "LAU-GRES")
)
_image_root = os.environ.get(
    "LAU_GRES_IMAGE_ROOT", str(_dataset_root / "images")
)
register_lau_gres(_data_root, _image_root)
