#!/usr/bin/env python
# --------------------------------------------------------
# Positive samples generator
# Testing why a perfect overlap does not exist
# --------------------------------------------------------
import _init_paths
import numpy as np
import os
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
from fast_rcnn.test import im_detect
from utils.timer import Timer
import caffe
import numpy as np
import cv2
from scipy.spatial.distance import cdist

class pos_sample_generator(object):
    def __init__(self,net,dataset_1,dataset_2,dataset_j,dim=4096):
        self.dim = dim
        self.layer = 'fc7'
        self.net = net
        self.dataset_1  = dataset_1 
        self.dataset_2  = dataset_2
        self.dataset_j  = dataset_j
        self.joint_map = self._get_joint_map()

    def _get_joint_map(self):
        classes_1 = self.dataset_1.classes
        classes_2 = self.dataset_2.classes
        classes_j = self.dataset_j.classes
        joint_map = [(0,0)] 
        for j in xrange(1,len(classes_j)):
            joint_reg = classes_j[j].split('-')
            c_1 = classes_1.index(joint_reg[0])
            c_2 = classes_2.index(joint_reg[1])
            joint_map.append((c_1,c_2))
        return joint_map
  
    def _get_pos_counts(self,imdb):
        counts = np.zeros((imdb.num_classes), dtype=np.int)
        roidb = imdb.roidb
        for i in xrange(len(roidb)):
            for j in xrange(1,imdb.num_classes):
                I = np.where(roidb[i]['gt_classes'] == j)[0]
                counts[j] += len(I)

        for j in xrange(1,imdb.num_classes):
            print('class {:s} has {:d} positives'.
                  format(imdb.classes[j], counts[j]))

        return counts

    def _get_image_idx(self,dataset,im_path):
        num_images = len(dataset.roidb)
        idx = [i for i in range(num_images) if os.path.basename(dataset.image_path_at(i))==os.path.basename(im_path)]
        assert len(idx)<=1,'Duplicate appearance of image {} in dataset'.format(os.path.basename(im_path))
        if len(idx)==0: return None
        else          : return idx[0]
    
    def _collect_im_pos(self,im,rois):
        if rois['flipped']:
               im = im[:, ::-1, :]
        gt_inds = np.where(rois['gt_classes'] > 0)[0]
        gt_boxes = rois['boxes'][gt_inds].astype(np.float)
        _, __ = im_detect(self.net, im, gt_boxes)
        feat = self.net.blobs[self.layer].data
        im_classes = np.unique(rois['gt_classes'][gt_inds])
        im_feat = []
        for j in im_classes:
            cls_inds = np.where(rois['gt_classes'][gt_inds] == j)[0]
            cls_feat = feat[cls_inds, :]
            im_feat.append(cls_feat)
      
        return (im_feat,im_classes)

    def _append_pos(self,pos,feat, cls):
        for idx in range(len(cls)):
            c = cls[idx]
            pos[c] = np.vstack((pos[c],feat[idx]))
        return pos

    def _verify_feature_intersection(self,feat_1, cls_1,feat_2, cls_2,feat_j, cls_j):
        for idx_j in range(len(cls_j)):
            class_j = cls_j[idx_j]
            feat_j_cls = feat_j[idx_j]
            #get candidate matches
            feat_1_cls = feat_1[np.argwhere(cls_1==self.joint_map[class_j][0]).ravel()[0]]
            feat_2_cls = feat_2[np.argwhere(cls_2==self.joint_map[class_j][1]).ravel()[0]]
            #check vs feat1
            dist = cdist(feat_j_cls,feat_1_cls)
            if max(np.amin(dist,axis=1))>0: import pdb; pdb.set_trace()
            #check vs feat2
            dist = cdist(feat_j_cls,feat_2_cls)
            if max(np.amin(dist,axis=1))>0: import pdb; pdb.set_trace()



    def collect_positives(self):
        #TODO-may this can be removed
        '''
        #count positives
        n_pos_1 = self._get_pos_counts(self.dataset_1)
        n_pos_2 = self._get_pos_counts(self.dataset_2)
        n_pos_j = self._get_pos_counts(self.dataset_j)
        #allocate positives
        pos_1 = [np.zeros((n, self.dim), dtype=np.float32) for n in n_pos_1]
        pos_2 = [np.zeros((n, self.dim), dtype=np.float32) for n in n_pos_2]
        pos_j = [np.zeros((n, self.dim), dtype=np.float32) for n in n_pos_j]
        '''
        pos_1 = [np.zeros((0, self.dim), dtype=np.float32) for i in xrange(self.dataset_1.num_classes)]
        pos_2 = [np.zeros((0, self.dim), dtype=np.float32) for i in xrange(self.dataset_2.num_classes)]
        pos_j = [np.zeros((0, self.dim), dtype=np.float32) for i in xrange(self.dataset_j.num_classes)]
        #load roidbs
        roidb_1 = self.dataset_1.roidb
        roidb_2 = self.dataset_2.roidb
        roidb_j = self.dataset_j.roidb
        ####################run over datasets######################3
        checked_images = []
        #run on dataset_1
        for idx_1 in range(len(roidb_1)):
            #get indices from other datasets
            im_path = self.dataset_1.image_path_at(idx_1)
            idx_2   = self._get_image_idx(self.dataset_2,im_path)
            idx_j   = self._get_image_idx(self.dataset_j,im_path)
            print 'image {}: idx1 = {},idx2 = {},idxj = {},'.format(os.path.basename(im_path),idx_1,idx_2,idx_j)
            #read image
            im = cv2.imread(im_path)
            #collect positives from each dataset
            feat_1, cls_1 = self._collect_im_pos(im,roidb_1[idx_1])
            pos_1 = self._append_pos(pos_1,feat_1, cls_1)
            if idx_2 is not None:
                feat_2, cls_2 = self._collect_im_pos(im,roidb_2[idx_2])
                pos_2 = self._append_pos(pos_2,feat_2, cls_2)
            if idx_j is not None:
                feat_j, cls_j = self._collect_im_pos(im,roidb_j[idx_j])
                pos_j = self._append_pos(pos_j,feat_j, cls_j)
            #verify intersection
            assert (idx_j is not None and idx_2 is not None) or idx_j is None, 'Datasets miss-alignment in image {}'.format(os.path.basename(im_path)) 
            if idx_j is not None:
                self._verify_feature_intersection(feat_1, cls_1,feat_2, cls_2,feat_j, cls_j)
            #add to image list to avoid second pass
            checked_images.append(os.path.basename(im_path))
            print 'Extracted positive features from image {}/{}'.format(idx_1+1,len(roidb_1))
        import pdb; pdb.set_trace()
        #run on dataset_2
        for idx_2 in range(len(roidb_2)):
            #get indices from other datasets
            im_path = self.dataset_2.image_path_at(idx_2)
            #check if image already was processed
            if os.path.basename(im_path) in checked_images: continue
            idx_1   = self._get_image_idx(self.dataset_1,im_path)
            idx_j   = self._get_image_idx(self.dataset_j,im_path)
            #idx_1 and idx_j should be None
            assert idx_j is None and idx_1 is None,'Datasets miss-alignment in image {}'.format(os.path.basename(im_path)) 
            #read image
            im = cv2.imread(im_path)
            #collect positives from each dataset
            feat_2, cls_2 = self._collect_im_pos(im,roidb_2[idx_2])
            pos_2 = self._append_pos(pos_2,feat_2, cls_2)
            #add to image list to avoid second pass
            checked_images.append(os.path.basename(im_path))
            print 'Extracted positive features from image {}/{}'.format(idx_2+1,len(roidb_2))
        import pdb; pdb.set_trace()

        


if __name__ == '__main__':
    #######params#######
    gpu_id = 4
    
    #configuration
    cfg.DEDUP_BOXES = 0
    cfg.TEST.SVM = True
    cfg.TRAIN.USE_FLIPPED = False
    #load some caffe net for feature extraction
    caffe.set_device(gpu_id)
    net = caffe.Net('/home/guyrose3/fast-rcnn/models/costumized/CaffeNet_146_classes/test.prototxt', 'data/imagenet_models/CaffeNet.v2.caffemodel', caffe.TEST)
    #load databases
    imdb_obj    = get_imdb('sg_dataset_objects_266.train')
    imdb_att   = get_imdb('sg_dataset_attributes_145.train')
    imdb_joint = get_imdb('sg_dataset_objects_attributes_2295.train')
    #init generator
    pos_gen = pos_sample_generator(net,imdb_obj,imdb_att,imdb_joint)
    #get positive examples
    pos_gen.collect_positives()














         

