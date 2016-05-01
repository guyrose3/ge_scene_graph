#!/usr/bin/env python

# --------------------------------------------------------
# Written by Guy Rosenthal
# --------------------------------------------------------
"""Test a Fast R-CNN network on an image database."""
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../','init_paths.py'))
import _init_paths
from detector.obj_detector import detector
from fast_rcnn.performance_evaluator import performance_evaluator as pe
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a Fast R-CNN network on a given database')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='disney_dataset', type=str)
    parser.add_argument('--datapath', dest='data_path',
                        help='dataset to test',
                        default=os.path.join(paths.DATASETS_BASE,'DisneyCharacter'), type=str)
    parser.add_argument('--set', dest='image_set',
                        help='image set name. train or test',
                        default='test', type=str)
    parser.add_argument('--conf', dest='conf_th',
                        help='confidence threshold for confusion matrix generation',
                        default='0.3', type=float)
    parser.add_argument('--prec', dest='prec_th',
                        help='precision threshold for MaP calculation',
                        default='0', type=float)
    parser.add_argument('--svm', dest='use_svm',
                        help='flag for using svm',
                        default=False, type=bool)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def check_args(args):
    """
    Check existance of all files
    """
    assert os.path.exists(args.prototxt)   , 'Prototxt file {} does not exist.'.format(args.prototxt)
    assert os.path.exists(args.caffemodel) , 'Caffemodel file {} does not exist.'.format(args.caffemodel)
    assert os.path.exists(os.path.join(args.data_path,'labels.txt'))   , 'No labels.txt file found in data path.'

if __name__ == '__main__':
    args = parse_args()

    check_args(args)

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    #set detector
    class_path = os.path.join(args.data_path,'labels.txt')
    det = detector(args.prototxt,args.caffemodel,class_path,gpu_id=args.gpu_id,use_svm=args.use_svm)
    #set evaluator
    d = pe(args.imdb_name,args.image_set, args.data_path)
    d.set_detector(det)
    d.get_detections()
    d.evaluate(confusion_th=args.conf_th)
    
    #debug-visualize bounding boxes
    d.detector.vis_detections(0.3,out_dir = os.path.join(paths.DATASETS_BASE,'DisneyCharacter/detection_examples_0.3'))
    #d.vis_gt_boxes()
