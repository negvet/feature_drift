import math
from copy import deepcopy

import cv2
import numpy as np
import torch

from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose

from mmdet.apis import inference_detector
from mmdet.models.detectors.faster_rcnn import FasterRCNN


def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask


def mask_image(image, mask):
    masked = image * np.dstack((mask, mask, mask))
    return masked


def iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)


def generate_saliency_map_drise(model,
                                data,
                                target_class_index,
                                target_box,
                                prob_thresh=0.5,
                                grid_size=(16, 16),
                                n_masks=5000,
                                seed=0):
    np.random.seed(seed)
    image_copy = deepcopy(data)
    image_h, image_w = image_copy.shape[:2]
    res = np.zeros((image_h, image_w), dtype=np.float32)
    for mask_id in range(n_masks):
        mask = generate_mask(image_size=(image_w, image_h),
                             grid_size=grid_size,
                             prob_thresh=prob_thresh)
        image_copy_masked = mask_image(image_copy, mask)
        image_copy_masked = image_copy_masked.astype(np.uint8)
        out = inference_detector(model, image_copy_masked)
        boxes_scores_labels = zip(out.pred_instances.bboxes.cpu(),
                                  out.pred_instances.scores.cpu(),
                                  out.pred_instances.labels.cpu().numpy())
        score = max([
            iou(target_box, box) * score if label == target_class_index else 0 for box, score, label in boxes_scores_labels
        ], default=0)
        if isinstance(score, torch.Tensor):
            score = score.cpu().detach().numpy()
        res += mask * score
        if mask_id % 1000 == 0:
            print("mask num", mask_id)
    return res / res.max()


def generate_mask_fm(backbone_out, grid_size, prob_thresh):
    target_scale_id = len(backbone_out) // 2
    image_w, image_h = backbone_out[target_scale_id].shape[2:]
    max_grid_size = min(image_h, grid_size[0])
    grid_size = max_grid_size, max_grid_size
    mask = generate_mask(image_size=(image_w, image_h),
                         grid_size=grid_size,
                         prob_thresh=prob_thresh)
    mask_fm = []
    for t in backbone_out:
        _, _, fm_h, fm_w = t.size()
        mask_fm.append(
            cv2.resize(mask, (fm_w, fm_h), interpolation=cv2.INTER_LINEAR).astype(
                np.float32))
    return mask_fm, mask


def generate_saliency_map_drise_feature_map_space(model,
                                                  data,
                                                  target_class_index,
                                                  target_box,
                                                  prob_thresh=0.5,
                                                  grid_size=(16, 16),
                                                  n_masks=5000,
                                                  seed=0):
    np.random.seed(seed)
    ori_h, ori_w, _ = data.shape

    cfg = model.cfg
    cfg = cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline)

    data_ = dict(img=data, img_id=0)

    # build the data pipeline
    data_ = test_pipeline(data_)
    data_['inputs'] = [data_['inputs']]
    data_['data_samples'] = [data_['data_samples']]
    data = model.data_preprocessor(data_, False)

    backbone_out = model.backbone(data['inputs'])
    if model.with_neck:
        backbone_out = model.neck(backbone_out)

    saliency_map = None
    for mask_id in range(n_masks):
        mask_per_fm, mask = generate_mask_fm(backbone_out, grid_size, prob_thresh)

        backbone_out_clone = [torch.clone(t) for t in backbone_out]
        backbone_out_clone = [backbone_out_clone[i] * torch.tensor(mask_per_fm[i]).cuda() for i in
                              range(len(backbone_out_clone))]

        if isinstance(model, FasterRCNN):
            rpn_results_list = model.rpn_head.predict(backbone_out_clone, data_['data_samples'], rescale=False)
            results_list = model.roi_head.predict(backbone_out_clone, rpn_results_list, data_['data_samples'], rescale=True)
            result = results_list[0]
        else:
            result = model.bbox_head.predict(backbone_out_clone, data_['data_samples'], rescale=True)[0]

        boxes_scores_labels = zip(result.bboxes.cpu(),
                                  result.scores.cpu(),
                                  result.labels.cpu().numpy())
        score = max([iou(target_box, box.detach().numpy()) * score if label == target_class_index else 0 for box, score, label in boxes_scores_labels], default=0)

        if isinstance(score, torch.Tensor):
            score = score.cpu().detach().numpy()

        if saliency_map is None:
            saliency_map = mask * score
        else:
            saliency_map += mask * score
        if mask_id % 1000 == 0:
            print("mask num", mask_id)

    saliency_map = saliency_map - saliency_map.min()
    saliency_map /= saliency_map.max()
    saliency_map = cv2.resize(saliency_map, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
    return saliency_map
