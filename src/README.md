# D-RISE implementation
Implementation is based on [D-RISE](https://arxiv.org/pdf/2006.03204.pdf) paper.
Random mask resolution is (16, 16). Number of masks is 10000.
Single target bbox is picked per image, if confident prediction is available.

Original work generates masks to sample input image. 
Current approach also generates random masks to sample from the feature map space. 
Masks are generated for middle FPN scale (`target_scale_id = len(backbone_out) // 2`) 
and then rescaled to other upstream and downstream scale levels. 
This might be not the best approach, but it produces decent baseline. 
See [d_rise](d_rise.py) for more details.

Performance of the implementation is far from being optimal, but enough to perform initial investigation.

# Installation
For installation follow instructions in mmdetection [get_started](https://mmdetection.readthedocs.io/en/latest/get_started.html).
To reproduce the results, use versions specified in requirements.txt (optional).

# Usage
```python
python src/run.py path_to_configs path_to_checkpoints path_to_data path_to_output
```