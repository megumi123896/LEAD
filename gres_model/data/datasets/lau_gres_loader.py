"""Loader for the LAU-GRES-derived GRefCOCO-style multi-target dataset.

This file is intentionally separate from the original RefCOCO/GRefCOCO loaders.
"""

import contextlib
import copy
import io
import json
import logging
import os

import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

logger = logging.getLogger(__name__)


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _lau_gres_category(category_id):
    """Map LAU-GRES category ids 1..8 to contiguous count ids 0..7.

    0 people, 1 car, 2 motor, 3 bicycle, 4 tricycle, 5 truck, 6 bus, 7 boat.
    """

    category_id = int(category_id)
    if not 1 <= category_id <= 8:
        raise ValueError(f"Unexpected LAU-GRES category id: {category_id}")
    return category_id - 1


def _normalize_segmentation(segmentation, height, width):
    if isinstance(segmentation, list) and len(segmentation) == 1 and isinstance(segmentation[0], dict):
        segmentation = segmentation[0]

    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            return mask_util.frPyObjects(segmentation, height, width)
        return segmentation

    if isinstance(segmentation, list):
        polygons = [
            poly for poly in segmentation
            if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0
        ]
        return polygons

    return None


def load_lau_gres_json(data_root, split, image_root, extra_annotation_keys=None):
    logger.info("Loading LAU-GRES split=%s from %s", split, data_root)
    timer = Timer()

    data_root = PathManager.get_local_path(data_root)
    instance_path = os.path.join(data_root, "instances.json")
    ref_path = os.path.join(data_root, "LAU-GRES_grefs_full.json")

    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"instances.json not found: {instance_path}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"LAU-GRES_grefs_full.json not found: {ref_path}")

    with contextlib.redirect_stdout(io.StringIO()):
        instances_json = json.load(open(instance_path, encoding="utf-8"))
        refs = json.load(open(ref_path, encoding="utf-8"))

    images = {img["id"]: img for img in instances_json["images"]}
    annotations = {ann["id"]: ann for ann in instances_json["annotations"]}

    ann_keys = ["iscrowd", "bbox", "area"] + (extra_annotation_keys or [])
    dataset_dicts = []
    ann_cache = {}
    skipped_missing_ann = 0
    skipped_bad_mask = 0

    for ref in refs:
        if ref.get("split") != split:
            continue

        image_id = ref["image_id"]
        img = images[image_id]
        record = {
            "source": "lau_gres",
            "file_name": os.path.join(image_root, ref.get("file_name", img["file_name"])),
            "height": img["height"],
            "width": img["width"],
            "image_id": image_id,
            "ref_id": ref["ref_id"],
            "empty": bool(ref.get("no_target", False)),
        }

        ann_ids = _as_list(ref.get("ann_id", ref.get("ann_ids")))
        category_ids = _as_list(ref.get("category_id", ref.get("category_ids")))

        objs = []
        if record["empty"]:
            # No-target samples are supervised only through the empty/nt_label
            # branch, matching GRefCOCO. They have no count target classes.
            record["category_id"] = []
        else:
            record["category_id"] = [_lau_gres_category(cat) for cat in category_ids]
            for ann_id in ann_ids:
                if ann_id not in annotations:
                    skipped_missing_ann += 1
                    continue
                if ann_id in ann_cache:
                    objs.append(copy.deepcopy(ann_cache[ann_id]))
                    continue

                anno = annotations[ann_id]
                obj = {key: anno[key] for key in ann_keys if key in anno}
                obj["bbox"] = anno["bbox"]
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                obj["iscrowd"] = anno.get("iscrowd", 0)
                # instances.json uses categories_id.
                raw_cat = anno.get("category_id", anno.get("categories_id"))
                obj["category_id"] = int(raw_cat) - 1
                obj["empty"] = False

                segmentation = _normalize_segmentation(
                    anno.get("segmentation"), img["height"], img["width"]
                )
                if not segmentation:
                    skipped_bad_mask += 1
                    continue
                obj["segmentation"] = segmentation
                ann_cache[ann_id] = copy.deepcopy(obj)
                objs.append(obj)

        if not record["empty"] and len(objs) == 0:
            # A targeted sample without valid masks cannot train/evaluate
            # segmentation, so skip it loudly but safely.
            skipped_bad_mask += 1
            continue

        record["annotations"] = objs
        for sent in ref.get("sentences", []):
            ref_record = copy.deepcopy(record)
            ref_record["sentence"] = {
                "raw": sent.get("raw", sent.get("sent", "")),
                "sent": sent.get("sent", sent.get("raw", "")),
                "sent_id": sent.get("sent_id"),
                "tokens": sent.get("tokens", []),
                "ref_id": ref["ref_id"],
            }
            dataset_dicts.append(ref_record)

    if timer.seconds() > 1:
        logger.info("Loading LAU-GRES %s took %.2f seconds.", split, timer.seconds())
    logger.info(
        "Loaded %d LAU-GRES sentence samples for split=%s. skipped_missing_ann=%d skipped_bad_mask=%d",
        len(dataset_dicts),
        split,
        skipped_missing_ann,
        skipped_bad_mask,
    )
    return dataset_dicts
