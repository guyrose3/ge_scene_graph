# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.abspath(__file__))
#base paths-should be changed to match local paths
FAST_RCNN_BASE = '/home/guy/thesis/gen_scene_graph/fast-rcnn'
DATASETS_BASE = '/home/guy/thesis/gen_scene_graph/data_tmp/'
CAFFE_DETECTORS_BASE = osp.join(this_dir,'caffe_detectors')
USE_MATLAB_SELECTIVE_SEARCH = False

# Add caffe to PYTHONPATH
caffe_path = osp.join(FAST_RCNN_BASE, 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(FAST_RCNN_BASE, 'lib')
add_path(lib_path)

# Add fast-rcnn to PYTHONPATH
ss_path = osp.join(FAST_RCNN_BASE)
add_path(ss_path)
