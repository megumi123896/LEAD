"""Evaluator for LAU-GRES, including mIoU/oIoU and no-target accuracy."""

import itertools   
import json
import os
from collections import OrderedDict   

import numpy as np
import torch
from detectron2.evaluation.evaluator import DatasetEvaluator    
from detectron2.utils.comm import all_gather, is_main_process, synchronize    
from detectron2.utils.file_io import PathManager


def _count_to_categories(count_result):            
    count_result = torch.round(count_result)   
    categories = []
    for index, count in enumerate(count_result):
        categories += [index] * int(count)
    return categories


def _count_to_vector(category_ids, num_classes):    
    count_vector = [0] * num_classes
    for category_id in category_ids:
        if 0 <= int(category_id) < num_classes:
            count_vector[int(category_id)] += 1
    return count_vector


class LAUGRESEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None):
        self.dataset_name = dataset_name
        self.distributed = distributed
        self.output_dir = output_dir
        self.cpu = torch.device("cpu")

    def reset(self):     
        self.predictions = []

    def process(self, inputs, outputs):  
        for model_input, output in zip(inputs, outputs):    
            pred_mask = output["ref_seg"].argmax(dim=0).to(self.cpu).numpy().astype(bool)     
            gt_mask = (    
                model_input["gt_mask_merged"]
                .to(self.cpu)
                .numpy()
                .astype(bool)
                .squeeze(0)
            )
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())     
            union = int(np.logical_or(pred_mask, gt_mask).sum())         

            pred_nt = bool(output["nt_label"].argmax(dim=0).bool().to(self.cpu))   
            gt_nt = bool(model_input.get("empty", False))   
            count_pred_tensor = output["count_pred"].detach().to(self.cpu)     
            pred_count_vector = count_pred_tensor.tolist()    
            pred_count_rounded_vector = torch.round(count_pred_tensor).int().tolist()   
            pred_count = _count_to_categories(count_pred_tensor)      
            gt_count = sorted(model_input.get("category_id", []))    
            gt_count_vector = _count_to_vector(gt_count, len(pred_count_vector))    

            if gt_nt:   
                sample_iou = 1.0 if pred_nt else 0.0
            else:
                sample_iou = 0.0 if union == 0 else intersection / union

            self.predictions.append(    
                {
                    "image_id": model_input["image_id"],
                    "file_name": model_input["file_name"],
                    "ref_id": model_input["sentence"]["ref_id"],
                    "sent_id": model_input["sentence"]["sent_id"],
                    "sentence": model_input["sentence"]["raw"],
                    "gt_nt": gt_nt,
                    "pred_nt": pred_nt,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "gt_count_vector": gt_count_vector,
                    "pred_count_vector": pred_count_vector,
                    "pred_count_rounded_vector": pred_count_rounded_vector,
                    "intersection": intersection,
                    "union": union,
                    "iou": sample_iou,
                }
            )

    def evaluate(self):    
        predictions = self.predictions
        if self.distributed:    
            synchronize()
            predictions = list(itertools.chain(*all_gather(predictions)))
            if not is_main_process():
                return None

        total = len(predictions)     
        if total == 0:
            return OrderedDict([("lau_gres", {})])

        target_predictions = [p for p in predictions if not p["gt_nt"]]   
        total_target = len(target_predictions)

        target_iou = 0.0   
        target_intersection = 0    
        target_union = 0   
        generalized_iou = 0.0   
        generalized_intersection = 0
        generalized_union = 0
        count_correct = 0    
        positives = []    
        negatives = []    

        for item in predictions:
            if item["gt_nt"]:   
                negatives.append(item)
                if item["pred_nt"]:
                    # Original LEAD treats a correctly predicted no-target
                    # sample as IoU=1 for gIoU, but it contributes no pixels
                    # to cIoU.
                    generalized_iou += 1.0      
                else:
                    # False negative no-target: penalize cumulative IoU by
                    # adding the predicted foreground union with zero
                    # intersection, matching the original evaluator.
                    generalized_union += item["union"]   

                if item["pred_count"] == item["gt_count"]:   
                    count_correct += 1
                continue

            positives.append(item)
            intersection = 0 if item["pred_nt"] else item["intersection"]
            union = item["union"]
            sample_iou = 0.0 if union == 0 else intersection / union

            target_iou += sample_iou
            target_intersection += intersection
            target_union += union
            generalized_iou += sample_iou
            generalized_intersection += intersection
            generalized_union += union
            if item["pred_count"] == item["gt_count"]:
                count_correct += 1

        results = {
            # Conventional target-only segmentation metrics.
            "mIoU": 0.0 if total_target == 0 else 100.0 * target_iou / total_target,
            "oIoU": 0.0 if target_union == 0 else 100.0 * target_intersection / target_union,
            # GRef/LEAD-style metrics with no-target decisions included.
            "gIoU": 100.0 * generalized_iou / total,
            "cIoU": 0.0 if generalized_union == 0 else 100.0 * generalized_intersection / generalized_union,
            "count_acc": 100.0 * count_correct / total,
        }
        for threshold in (0.5, 0.7, 0.8, 0.9):
            results[f"Pr@{threshold:.1f}"] = (
                0.0
                if total_target == 0
                else 100.0
                * sum(item["iou"] >= threshold for item in target_predictions)
                / total_target
            )

        results["T_acc"] = (
            0.0
            if not positives
            else 100.0 * sum(not p["pred_nt"] for p in positives) / len(positives)
        )
        results["N_acc"] = (
            0.0
            if not negatives
            else 100.0 * sum(p["pred_nt"] for p in negatives) / len(negatives)
        )
        # Lowercase alias, convenient for scripts that look for n_acc.
        results["n_acc"] = results["N_acc"]

        if self.output_dir:
            PathManager.mkdirs(self.output_dir)
            with PathManager.open(
                os.path.join(self.output_dir, f"{self.dataset_name}_results.json"), "w"
            ) as handle:
                json.dump(results, handle, indent=2)
            with PathManager.open(
                os.path.join(self.output_dir, f"{self.dataset_name}_detailed_results.json"),
                "w",
            ) as handle:
                json.dump(predictions, handle, indent=2)

        return OrderedDict([("lau_gres", results)])
