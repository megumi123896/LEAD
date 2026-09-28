"""Mapper for LAU-GRES RLE masks and no-target samples."""

import copy

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances

from .refcoco_mapper import RefCOCOMapper


class LAUGRESMapper(RefCOCOMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = ~transforms.apply_segmentation(padding_mask).astype(bool)
        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["padding_mask"] = torch.as_tensor(
            np.ascontiguousarray(padding_mask)
        )

        empty = dataset_dict.get("empty", False)
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0 and not obj.get("empty", False)
        ]

        if annos:
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format="bitmask"
            )
            gt_masks = instances.gt_masks.tensor.to(torch.uint8)
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        else:
            instances = Instances(image_shape)
            instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
            instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            instances.gt_masks = BitMasks(
                torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.bool)
            )
            gt_masks = instances.gt_masks.tensor.to(torch.uint8)

        if self.is_train:
            dataset_dict["instances"] = instances
        else:
            dataset_dict["gt_mask"] = gt_masks

        dataset_dict["empty"] = empty
        dataset_dict["gt_mask_merged"] = self._merge_masks(gt_masks)

        sentence_raw = dataset_dict["sentence"]["raw"]
        input_ids = self.tokenizer.encode(
            text=sentence_raw, add_special_tokens=True
        )[: self.max_tokens]
        padded_input_ids = [0] * self.max_tokens
        attention_mask = [0] * self.max_tokens
        padded_input_ids[: len(input_ids)] = input_ids
        attention_mask[: len(input_ids)] = [1] * len(input_ids)
        dataset_dict["lang_tokens"] = torch.tensor(padded_input_ids).unsqueeze(0)
        dataset_dict["lang_mask"] = torch.tensor(attention_mask).unsqueeze(0)
        return dataset_dict
