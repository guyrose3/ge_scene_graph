#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Train post-hoc SVMs using the algorithm and hyper-parameters from
traditional R-CNN.
"""
from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
from fast_rcnn.test import im_detect
from utils.timer import Timer
import caffe
import argparse
import numpy as np
import cv2
import cPickle as pickle
import os, sys

def unique_boxes(boxes):
    unique_boxes = np.array(boxes[0],dtype=np.int32)
    for i in range(1,len(boxes)):
        b = np.array(boxes[i],dtype=np.int32)
        sum_dim = len(unique_boxes.shape)-1
        if ((np.abs(unique_boxes - b)).sum(axis=sum_dim)).min()>0:
            unique_boxes = np.vstack((unique_boxes,b))
    return unique_boxes.astype(np.uint16)


def get_image_info(imdbs,num_images,image_set):
    #make sure all images appears
    paths1 = [os.path.basename(imdbs[0].image_path_at(i)) for i in range(imdbs[0].num_images)]
    paths2 = [os.path.basename(imdbs[1].image_path_at(i)) for i in range(imdbs[1].num_images)]
    image_list = list(set(paths1) | set(paths2)) 
    assert len(image_list)==num_images, 'not all images appearing in 2 databases'
    paths = [paths1,paths2]
    #get index for each image
    indices = [(0,0)] * len(image_list)
    for i in range(len(image_list)):
        p = image_list[i]
        if p in paths[0]: indices[i] = (0,paths[0].index(p))
        else            : indices[i] = (1,paths[1].index(p))
    #gt - only for train
    if image_set=='train':
        print 'Start fetching gt boxes'
        gt_db = []
        gts = [imdbs[0].gt_roidb(),imdbs[1].gt_roidb()]
        for i in range(len(image_list)):
            boxes = []
            p = image_list[i]
            if p in paths[0]:
                im_ind = paths[0].index(p)
                boxes += gts[0][im_ind]['boxes'].tolist() 
            if p in paths[1]:
                im_ind = paths[1].index(p)
                boxes += gts[1][im_ind]['boxes'].tolist()
            gt_db.append(unique_boxes(boxes))
        print 'Done'
    #rois
    print 'Start fetching ROI boxes'
    roi_db = []
    rois = [imdbs[0].roidb,imdbs[1].roidb]
    for i in range(len(image_list)):
        p = image_list[i]
        if p in paths[0]:
            im_ind = paths[0].index(p)
            boxes = rois[0][im_ind]['boxes'] 
        elif p in paths[1]:
            im_ind = paths[1].index(p)
            boxes = rois[1][im_ind]['boxes']
        roi_db.append(boxes)
    print 'Done'
    if image_set=='train':
        return {'image_name' : image_list,'roi' : roi_db, 'gt' : gt_db}
    else:
        return {'image_name' : image_list,'roi' : roi_db}

def build_feature_db(net,images_info,imdbs,out_obj):
    paths1 = [os.path.basename(imdbs[0].image_path_at(i)) for i in range(imdbs[0].num_images)]
    paths2 = [os.path.basename(imdbs[1].image_path_at(i)) for i in range(imdbs[1].num_images)]
    paths = [paths1,paths2]
    #im_db = []
    _t = Timer()
    for i in range(len(images_info['image_name'])):
        print 'caching features for image {:d}/{:d}'.format(i+1,len(images_info['image_name']))
        _t.tic()
        if   images_info['image_name'][i] in paths[0]:
            im = cv2.imread(imdbs[0].image_path_at(paths[0].index(images_info['image_name'][i])))
        elif images_info['image_name'][i] in paths[1]:
            im = cv2.imread(imdbs[1].image_path_at(paths[1].index(images_info['image_name'][i])))
        print 'Done running NN'
        #gt features
        if 'gt' in images_info.keys():
            scores, boxes = im_detect(net,im,images_info['gt'][i])
            feat_pos = net.blobs['fc7'].data
        #roi features
        scores, boxes = im_detect(net,im,images_info['roi'][i])
        feat_neg = net.blobs['fc7'].data
        print 'Done extracting features from fc7'
        #generate image db
        
        if 'gt' in images_info.keys():
            im_reg = {'name' : images_info['image_name'][i], 'roi_boxes' : images_info['roi'][i], 'roi_features' : feat_neg,'gt_boxes' : images_info['gt'][i], 'gt_features' : feat_pos}
        else:
            im_reg = {'name' : images_info['image_name'][i], 'roi_boxes' : images_info['roi'][i], 'roi_features' : feat_neg}
        pickle.dump(im_reg,out_obj)
        _t.toc()
        #im_db.append(im_reg)
        print 'Done in {}'.format(_t.average_time)
    #return im_db
 


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='cache nn features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='/home/guyrose3/fast-rcnn/models/costumized/CaffeNet_8_classes/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='data/imagenet_models/CaffeNet.v2.caffemodel', type=str)
    parser.add_argument('--set', dest='image_set',
                        help='image set name. train or test',
                        default='train', type=str)
    parser.add_argument('--outdir', dest='outdir',
                        help='Directory to write training results to',
                        default='/data06/guy/fast-rcnn/', type=str)

    '''
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    '''

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print('Called with args:')
    print(args)
    #config
    cfg.DEDUP_BOXES = 0
    cfg.TEST.SVM = True
    cfg.TRAIN.USE_FLIPPED = False
    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    #########net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    #load imdbs
    imdb_names = ['sg_dataset_objects_266'+ '.' + args.image_set,'sg_dataset_attributes_145'+ '.' + args.image_set]
    if   args.image_set=='train': num_images=4000
    elif args.image_set=='test' : num_images=1000
    imdbs = [get_imdb(n) for n in imdb_names]
    images_info = get_image_info(imdbs,num_images,args.image_set)
    out_obj = open(os.path.join(args.outdir,'sg_feature_cache.pkl'),'wb')
    #feature_db = build_feature_db(net,images_info,imdbs,out_obj)
    build_feature_db(net,images_info,imdbs,out_obj)
    import pdb; pdb.set_trace()
