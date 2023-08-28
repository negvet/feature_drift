# Feature drift of object detection models

Some CNN-based single-stage object detectors significantly change spatial locations of (salient) features, 
while moving from the input image space towards the feature map space. 
In particular, in the intermediate feature maps, activations might be specially shifted (in many cases towards the center of the object).

Image below highlights salient features in the input image space (middle column) and in the feature map space (right column).
Salient features are the ones that models use to detect a bbox of a 'person' class.

<div align="center">
    <img src="https://github.com/negvet/feature_drift/assets/17028475/a2bab030-e185-42ac-b25a-39d6e8f88703">
</div>

See more models and saliency maps in [examples](examples).

Such kind of behaviour is not the case for CNN-based classification architectures (Resnet, MobileNet, etc.).
That is the reason of well-developed CAM-based Explainable AI (XAI) methods for classifiers.
On the other side, many object detectors, while being designed to precisely estimate location of the objects, 
actually mess up spatial location of object features in the latent space.

## Which object detectors shift features?
Considering [examples](examples), it is possible to conclude that the following models (mostly) shift activations
towards the center of the object:
- [YOLOX](https://arxiv.org/pdf/2107.08430.pdf)
- [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)
- [SSD](https://arxiv.org/pdf/1512.02325.pdf)
- [RTMDet](https://arxiv.org/pdf/2212.07784.pdf)

While the following models mostly tend to preserve spacial location of the activations (although not in all cases):
- [ATSS](https://arxiv.org/pdf/1912.02424.pdf)
- [RetinaNet](https://arxiv.org/pdf/1708.02002v2.pdf)
- [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)

## Experiment methodology
XAI can be used to estimate which part of input (which features) makes the most contribution to the model prediction.
To visualize the most salient features, 
I applied [D-RISE](https://arxiv.org/pdf/2006.03204.pdf) to the input image and to the feature map (activation tensor). 
See implementation results in [REF].
Similar results might be obtained from the different approach - 
when visualizing normalized per-class slices of the raw classification head output (if available).

## Why this is happening?
Due to the loss design.
Only cells located in the proximity to the center of the object are getting gradient signal - 
IOU(target, prediction) is estimated, see iou_loss implementation in [mmdetection](https://github.com/open-mmlab/mmdetection/blob/f78af7785ada87f1ced75a2313746e4ba3149760/mmdet/models/losses/iou_loss.py#L47).
Therefore, the model explicitly learn to move features to the center of the object.
This is lees of an issue for e.g. two-stage detectors (Faster R-CNN) or RetinaNet, see [examples](examples).

## Which limitations does it bring?
It can limit anything that leverages internal network activations to recover spatial insights, e.g.:
- Activation-based XAI methods, which are somehow using feature maps, cannot always be directly used to explain object detectors. 
Due to fact that feature map space might not well preserve spatial information.
- Obtaining class activation map in YOLO paper (see Fig. 2 at [YOLO](https://arxiv.org/pdf/1506.02640.pdf) paper).
