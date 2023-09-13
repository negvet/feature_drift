import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

import mmcv
from mmdet.apis import init_detector, inference_detector

from d_rise import generate_saliency_map_drise, generate_saliency_map_drise_feature_map_space


MODELS = {
    "ssd512_coco": [
        'ssd/ssd512_coco.py',
        'ssd512_coco_20210803_022849-0a47a1ca.pth',
    ],
    "atss_r50_fpn_1x_coco":
    [
        'atss/atss_r50_fpn_1x_coco.py',
        'atss_r50_fpn_1x_coco_20200209-985f7bd0.pth',
    ],
    "rtmdet_tiny_8xb32-300e_coco":
    [
        'rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
        'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
    ],
    "yolox_l_8xb8-300e_coco":
    [
        'yolox/yolox_l_8xb8-300e_coco.py',
        'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    ],
    "yolov3_d53_8xb8-ms-416-273e_coco":
    [
        'yolo/yolov3_d53_8xb8-ms-416-273e_coco.py',
        'yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth',
    ],
    "retinanet_r50_fpn_2x_coco":
    [
        'retinanet/retinanet_r50_fpn_2x_coco.py',
        'retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth',
    ],
    "vfnet_r50_fpn_ms-2x_coco":
    [
        'vfnet/vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco.py',
        'vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth',
    ],
    "faster_rcnn_r50_fpn_2x_coco":
    [
        'faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py',
        'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
    ],
}
NUM_MASKS = 5000
IMG_LIMIT = 1


def plot(img, target_box, saliency_map, saliency_map_fm, full_path, save_name):
    # create figure
    fig = plt.figure()

    # setting values to rows and column variables
    rows = 1
    columns = 3

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(img[:, :, ::-1])
    plt.gca().add_patch(
        Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0],
                  target_box[3] - target_box[1],
                  linewidth=2, edgecolor='orange', facecolor='none'))
    plt.axis('off')
    plt.title("Original image.")

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(img[:, :, ::-1])
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    plt.gca().add_patch(
        Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0],
                  target_box[3] - target_box[1],
                  linewidth=2, edgecolor='orange', facecolor='none'))
    plt.axis('off')
    plt.title("Input image space.")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(img[:, :, ::-1])
    plt.imshow(saliency_map_fm, cmap='jet', alpha=0.5)
    plt.gca().add_patch(
        Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0],
                  target_box[3] - target_box[1],
                  linewidth=2, edgecolor='orange', facecolor='none'))
    plt.axis('off')
    plt.title("Feature map space.")
    # plt.show()
    print('save_name', save_name)
    fig.savefig(full_path, bbox_inches='tight', pad_inches=0.05)


def main():
    if len(sys.argv) != 5:
        raise RuntimeError(
            f"Usage: {sys.argv[0]} <path_to_configs> <path_to_checkpoints> <path_to_data> <path_to_output>"
        )

    path_to_configs = sys.argv[1]
    path_to_checkpoints = sys.argv[2]
    img_dir = sys.argv[3]  # one option is to go with coco val dataset
    output_dir = sys.argv[4]

    print("NUM_MASKS", NUM_MASKS, "IMG_LIMIT", IMG_LIMIT)

    img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    target_class_index = 0  # 0 - person  2 - car

    for model_name, conf in MODELS.items():
        config_file, checkpoint_file = conf
        config_file = os.path.join(path_to_configs, config_file)
        checkpoint_file = os.path.join(path_to_checkpoints, checkpoint_file)
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        img_counter = 0

        for img_name in img_names:
            img = mmcv.imread(os.path.join(img_dir, img_name))
            save_name = model_name + "_" + img_name[:-4] + '.png'
            full_path = os.path.join(output_dir, save_name)
            if os.path.exists(full_path):
                print(save_name + " already available, skip...")
                img_counter += 1
                continue

            result = inference_detector(model, img)
            target_box = None
            ori_h, ori_w, _ = img.shape
            for prediction in result.pred_instances:
                bbox, label, score = (prediction.bboxes[0].detach().cpu().numpy(), prediction.labels[0].detach().cpu().numpy(),
                                      prediction.scores[0].detach().cpu().numpy())
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                is_bbox_big = bbox_w / ori_w > 0.6 or bbox_h / ori_h > 0.6
                if label == target_class_index and score > 0.5 and is_bbox_big:
                    target_box = bbox
                    # print(bbox, label, score)
                    break

            if target_box is None:
                continue

            saliency_map = generate_saliency_map_drise(model,
                                                       img,
                                                       target_class_index,
                                                       target_box,
                                                       n_masks=NUM_MASKS)
            saliency_map_fm = generate_saliency_map_drise_feature_map_space(model,
                                                                            img,
                                                                            target_class_index,
                                                                            target_box,
                                                                            n_masks=NUM_MASKS)
            plot(img, target_box, saliency_map, saliency_map_fm, full_path, save_name)
            img_counter += 1

            if img_counter >= IMG_LIMIT:
                break


if __name__ == "__main__":
    main()
