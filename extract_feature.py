import argparse
import json
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import detectron2
from detectron2.config import CfgNode as CN
from detectron2.config.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers import batched_nms, nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))  # Root of detectron2
MIN_BOXES = 36
MAX_BOXES = 100


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_json", default="data/dataset_flickr8k.json", help=""
    )
    parser.add_argument(
        "--data_folder",
        default="data/Images",
        help="",
    )
    parser.add_argument("--out_folder", default="feature_out", help="")
    return parser.parse_args()


def build_model():
    # Build model and load weights.
    print("Load the Faster RCNN weight for ResNet101, pretrained on MS COCO detection.")
    cfg = get_cfg()
    # cfg.set_new_allowed(True)
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "pretrained/model_final_f6e8b1.pkl"

    # cfg.MODEL.SWINT = CN()
    # cfg.MODEL.SWINT.EMBED_DIM = 96
    # cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    # cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    # cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    # cfg.MODEL.SWINT.WINDOW_SIZE = 7
    # cfg.MODEL.SWINT.MLP_RATIO = 4
    # cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    # cfg.MODEL.SWINT.APE = False
    # cfg.MODEL.BACKBONE.FREEZE_AT = -1

    detector = DefaultPredictor(cfg)
    return detector


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def predict_boxes(predictions, proposals):
    cfg = get_cfg()
    if not len(proposals):
        return []
    box2box_transform = Box2BoxTransform(
        weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
    )
    _, proposal_deltas = predictions
    num_prop_per_image = [len(p) for p in proposals]
    proposal_boxes = [p.proposal_boxes for p in proposals]
    proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
    predict_boxes = box2box_transform.apply_deltas(
        proposal_deltas, proposal_boxes
    )  # Nx(KxB)
    return predict_boxes.split(num_prop_per_image)


def predict_probs(predictions, proposals):
    scores, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    probs = F.softmax(scores, dim=-1)
    return probs.split(num_inst_per_image, dim=0)


def doit(detector, raw_image):
    with torch.no_grad():
        # Preprocessing
        inputs = detector(raw_image)
        images = detector.model.preprocess_image([inputs])

        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(images, features, None)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        predictions, box_features = detector.model.roi_heads(
            images, features, proposals
        )
        feature_pooled = box_features.mean(
            dim=[2, 3]
        )  # (sum_proposals, 2048), pooled to 1x1

        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = predict_probs(predictions, proposals)
        boxes_list = predict_boxes(predictions, proposals)
        for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes,
                    probs,
                    image_size,
                    score_thresh=0.2,
                    nms_thresh=nms_thresh,
                    topk_per_image=MAX_BOXES,
                )
                if len(ids) >= MIN_BOXES:
                    break
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        num_preds_per_image = [len(p) for p in proposals]
        features_list = feature_pooled.split(
            num_preds_per_image
        )  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())

        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
            instances_list, [inputs], images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            raw_instances = detector_postprocess(instances, height, width)
            raw_instances_list.append(raw_instances)

        return raw_instances_list, roi_features_list


def gen_feature(args):
    detector = build_model()
    os.makedirs(os.path.join(args.out_folder), exist_ok=True)

    if not os.path.exists(args.input_json):
        print("Error: invalid json file.")

    imgs = json.load(open(args.input_json, "r"))
    imgs = imgs["images"]

    for i, img in enumerate(tqdm(imgs)):
        filename = img["filename"].split(".")[0]
        image_id = img["imgid"]

        im0 = cv2.imread(
            str(Path(os.path.join(args.data_folder, filename)).with_suffix(".jpg"))
        )
        h, w, _ = im0.shape

        instances_list, features_list = doit(detector, im0)
        instances = instances_list[0].to("cpu")
        features = features_list[0].to("cpu")

        num_objects = len(instances)
        image_bboxes = instances.pred_boxes.tensor.numpy()
        info = {
            "image_id": image_id,
            "image_h": h,
            "image_w": w,
            "objects_id": instances.pred_classes.numpy(),  # int64
            "objects_conf": instances.scores.numpy(),  # float32
            "attrs_id": np.zeros(num_objects, np.int64),  # int64
            "attrs_conf": np.zeros(num_objects, np.float32),  # float32
            "num_boxes": num_objects,
            # "boxes": base64.b64encode(instances.pred_boxes.tensor.numpy()).decode(),  # float32
            # "features": base64.b64encode(features.numpy()).decode()  # float32
        }
        output_file = os.path.join(args.out_folder, str(image_id))
        np.savez_compressed(
            output_file,
            features=features,
            bbox=image_bboxes,
            num_bbox=num_objects,
            image_h=h,
            image_w=w,
            info=info,
        )


if __name__ == "__main__":
    args = arguments()
    print(args)  # Print parsed arguments
    gen_feature(args)
