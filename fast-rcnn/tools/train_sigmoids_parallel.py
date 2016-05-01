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
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from fast_rcnn.test import im_detect
from utils.timer import Timer
import caffe
import argparse
import pprint
import numpy as np
import cv2
from sklearn.calibration import _sigmoid_calibration
from sklearn.cross_validation import train_test_split
import cPickle as pickle
import os, sys
from easydict import EasyDict as edict

class sigmoidFitter(object):
    """
    Trains post-hoc detection SVMs for all classes using the algorithm
    and hyper-parameters of traditional R-CNN.
    """

    def __init__(self, net, imdb,sigmoid_cfg,classes_to_train,fold_idx=None):
        self.cfg = sigmoid_cfg
        self.imdb = imdb
        self.net = net
        self.cls_prm = 'cls_score'
        self.classes_to_train = classes_to_train
        self.num_classes_to_train = len(classes_to_train)
        dim = net.params[self.cls_prm][0].data.shape[1]
        self.fold_idx = fold_idx
        self.trainers = [sigmoidClassFitter(imdb.classes[classes_to_train[i]], dim,self.cfg) for i in range(len(classes_to_train))]

    def _get_pos_counts(self):
        counts = np.zeros((len(self.num_classes_to_train)), dtype=np.int)
        roidb = self.imdb.roidb
        for i in xrange(len(roidb)):
            for j in xrange(self.num_classes_to_train):
                I = np.where(roidb[i]['gt_classes'] == self.classes_to_train[j])[0]
                counts[j] += len(I)

        for j in xrange(self.num_classes_to_train):
            print('class {:s} has {:d} positives'.
                  format(self.imdb.classes[self.classes_to_train[j]], counts[j]))

        return counts

    def gen_pos_examples(self):
        #TODO-change this function to support classes to train,num_classes_to_train
        #TODO-consider what to do with the flipped issue, should generate positives also through this function
        '''
        Collects positive examples from all images.
        stores them in a directory, one file for each class
        If folds are used, save along a fold index
        '''
        cache_dir = os.path.join(self.imdb.cache_path,'svm_pos')
        if os.path.exists(cache_dir): return
        #collect positive examples
        counts = self._get_pos_counts()
        for i in xrange(len(counts)):
            self.trainers[i].alloc_pos(counts[i])
        fold_idx = [[] for _ in xrange(self.imdb.num_classes)]
        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)
        for i in xrange(num_images):
            im = cv2.imread(self.imdb.image_path_at(i))
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            gt_inds = np.where(roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = roidb[i]['boxes'][gt_inds]
            _t.tic()
            scores, boxes = im_detect(self.net, im, gt_boxes)
            _t.toc()
            for j in xrange(1, self.imdb.num_classes):
                cls_inds = np.where(roidb[i]['gt_classes'][gt_inds] == j)[0]
                if len(cls_inds) > 0:
                    cls_feat = feat[cls_inds, :]
                    self.trainers[j].append_pos(cls_feat)
                    if imdb.k_fold:
                        fold_idx[j] += [imdb.folds[j,i]] * len(cls_inds)
            print 'get_pos_examples: {:d}/{:d} {:.3f}s'.format(i + 1, len(roidb), _t.average_time)
        print 'saving pos examples to cache'
        os.makedirs(cache_dir)
        for j in xrange(1, self.imdb.num_classes):
            cache_file = os.path.join(cache_dir,  'pos_class_{:d}.pkl'.format(j))
            pickle.dump([self.trainers[j].pos,np.asarray(fold_idx[j])],open(cache_file,'wb'))
        print 'Done saving pos examples to cache'


    def get_class_pos_examples(self,class_idx):
        cache_dir = os.path.join(self.imdb.cache_path,'svm_pos')
        cache_file = os.path.join(cache_dir,  'pos_class_{:d}.pkl'.format(self.classes_to_train[class_idx]))
        self.trainers[class_idx].pos,fold_array = pickle.load(open(cache_file,'rb'))
        #if not flip - take only half
        #TODO - need to make sure pos examples are always generated with flip
        if not self.cfg.USE_FLIPPED:
            tmp_pos_num = self.trainers[class_idx].pos.shape[0]
            assert tmp_pos_num % 2==0,'Must be even number of positive examples'
            assert self.trainers[class_idx].pos.shape[0]==len(fold_array),'pos example and fold array dims are different'
            self.trainers[class_idx].pos = self.trainers[class_idx].pos[:tmp_pos_num/2,:]
            fold_array = fold_array[:tmp_pos_num/2]
        if self.fold_idx is not None:
            keep_idx = np.argwhere(fold_array!=self.fold_idx).ravel()
            self.trainers[class_idx].pos = self.trainers[class_idx].pos[keep_idx,:]
        #generate positive scores for the fitting
        w = self.net.params[self.cls_prm][0].data[class_idx, :]
        b = self.net.params[self.cls_prm][1].data[class_idx]
        self.trainers[class_idx].pos = np.dot(self.trainers[class_idx].pos,w) + b
        self.trainers[class_idx].pos_cur = self.trainers[class_idx].pos.shape[0]
        

    def save_sigmoids(self,svm_models_file_name):
        params = [tr.sigmoid_params for tr in self.trainers]
        pickle.dump(params, open(svm_models_file_name, "wb"))

    def fit(self):
        print 'Start dataset sigmoid fitting'       
        # Pass over roidb to count num positives for each class
        #   a. Pre-allocate arrays for positive feature vectors
        # Pass over roidb, computing features for positives only
        # convert positive features to positive scores for sigmoid fitting
        self.gen_pos_examples()
        for j in xrange(self.num_classes_to_train):
            print 'loading positive examples for class {}'.format(self.classes_to_train[j])
            self.get_class_pos_examples(j)

        #sigmoid fitting
        print 'Start fitting sigmoids for classes {:d}-{:d}'.format(self.classes_to_train[0],self.classes_to_train[-1])
        num_neg = [self.cfg.NUM_SAMPLES_SIGMOID_FIT for _ in xrange(self.num_classes_to_train)]        
        sigmoid_neg = self.sample_random_neg_scores(num_neg)

        #calibration
        for j in xrange(self.num_classes_to_train):
            self.trainers[j].calibrate_detector(sigmoid_neg[j])
            self.trainers[j].fit_probability_sigmoid(sigmoid_neg[j])
 

    def sample_random_neg_scores(self,neg_num):
        neg = [[]] * self.num_classes_to_train
        for j in xrange(self.num_classes_to_train):
            neg[j] = np.zeros((0))

        done = [False] * self.num_classes_to_train
        for i in xrange(len(self.imdb.roidb)):
            if not (False in done): break
            im = cv2.imread(self.imdb.image_path_at(i))
            if self.imdb.roidb[i]['flipped']:im = im[:, ::-1, :]
            scores, boxes = im_detect(self.net, im, self.imdb.roidb[i]['boxes'])
            for j in xrange(self.num_classes_to_train):
                if neg[j].shape[0]>=neg_num[j]: 
                    done[j] = True
                    continue 
                neg_idx = np.where((self.imdb.roidb[i]['gt_overlaps'][:, self.classes_to_train[j]].toarray().ravel() < self.cfg.NEG_IOU_THRESH))[0]
                neg[j] = np.hstack((neg[j], scores[neg_idx, j].copy()))
        
        for j in xrange(self.num_classes_to_train):
            neg[j] = neg[j][:neg_num[j]]

        return neg
     
        

class sigmoidClassFitter(object):
    """Manages post-hoc SVM training for a single object class."""

    def __init__(self, cls, dim, sigmoid_cfg, pos_weight=1.0):
        self.cfg = sigmoid_cfg
        self.pos = np.zeros((0, dim), dtype=np.float32)
        self.neg = np.zeros((0, dim), dtype=np.float32)
        self.cls = cls
        self.pos_weight = self.cfg.BASE_POS_WEIGHT
        self.dim = dim
        self.pos_cur = 0
        self.sigmoid_params = []
        self.calib_params = []

    def alloc_pos(self, count):
        self.pos_cur = 0
        self.pos = np.zeros((count, self.dim), dtype=np.float32)

    def append_pos(self, feat):
        num = feat.shape[0]
        #try to catch exeption
        if self.pos[self.pos_cur:self.pos_cur + num, :].shape!=feat.shape:
            import pdb; pdb.set_trace()

        self.pos[self.pos_cur:self.pos_cur + num, :] = feat
        self.pos_cur += num

    def calibrate_detector(self,neg):
        '''
        Following calibration in "Seeing 3D chairs"(Aubry,2014):
        learn a linear mapping of detection score S' = a*S + b s.t:
        mean negative score is mapped to -1
        99% percentile is mapped to 0
        '''
        print('calibrating detector results to {} detector'.format(self.cls))
        neg = neg[np.argsort(neg)]
        mu_n = neg.mean()
        x = neg[int(len(neg)*0.99)]
        a = 1./(x-mu_n)
        b = x /(mu_n-x)
        self.calib_params = [a,b]     
        

    def fit_probability_sigmoid(self,neg):
        print('Fitting probablity sigmoid to {} detector'.format(self.cls))
        #apply calibration
        pos = self.pos * self.calib_params[0] + self.calib_params[1]
        neg =      neg * self.calib_params[0] + self.calib_params[1]

        num_pos = len(pos)
        num_neg = len(neg)
        X = np.hstack((pos,neg))
        y = np.hstack((np.ones(num_pos),-np.ones(num_neg)))
        #enforce max ratio between neg and pos example num
        if (num_neg/num_pos)>self.cfg.MAX_RATIO_NEG_POS:
            curr_pos_w = self.pos_weight * (num_neg / num_pos) * (1 / self.cfg.MAX_RATIO_NEG_POS)
            curr_neg_w = 1.
        else:
            curr_pos_w = self.pos_weight
            curr_neg_w = 1.
        weights = ([curr_pos_w] * num_pos) + ([curr_neg_w] * num_neg)
        A,B = _sigmoid_calibration(X, y, sample_weight=np.asarray(weights))
        #A,B = _sigmoid_calibration(X, y, sample_weight=None)
        #TODO-maybe these should be saved seperatly?
        self.sigmoid_params = [A,B,self.calib_params[0],self.calib_params[1]]
        #self.evaluate_sigmoid_match(X_test,y_test,A,B)
        print('Sigmoid parameters : {},{}'.format(self.sigmoid_params[0],self.sigmoid_params[1]))

    '''
    def evaluate_sigmoid_match(self,X_test,y_test,A,B):
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
        prob_pos = 1. / (1. + (np.exp(A * X_test + B)))
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_test.max())
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        print("SVC_sigmoid:")
        print("\tBrier: %1.3f" % (clf_score))
        fig = plt.figure(2, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")  
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (%1.3f)" % ("SVC_sigmoid", clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label="SVC_sigmoid",histtype="step", lw=2)
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        plt.tight_layout()
        plt.show()
    '''


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train SVMs (old skool)')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='disney_dataset', type=str)
    parser.add_argument('--set', dest='image_set',
                        help='image set name. train or test',
                        default='train', type=str)
    parser.add_argument('--outdir', dest='outdir',
                        help='Directory to write training results to',
                        default=None, type=str)
    parser.add_argument('--k_fold', dest='k_fold',
                        help='Optional k-fold support',
                        default=None, type=int)
    parser.add_argument('--first_class', dest='first_class',
                        help='First class idx to be trained',
                        default=None, type=int)
    parser.add_argument('--class_num', dest='class_num',
                        help='Max number of classes to be trained',
                        default=None, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def load_svm_coeffs(net,file_name,classes):
    print 'load svm coeffs : {:s}'.format(file_name)
    w,b = pickle.load(open(file_name, "rb"))
    net.params['cls_score'][0].data[:len(classes),:] = w[classes,:]
    net.params['cls_score'][1].data[:len(classes)]   = b[classes]

def get_sigmoid_configuration():
    config = edict()
    config.NUM_SAMPLES_SIGMOID_FIT = 50000
    config.NEG_IOU_THRESH = 0.3
    #TODO-decide how use these params
    config.USE_FLIPPED = False
    config.USE_INV_PROP_WEIGHTS = False
    config.BASE_POS_WEIGHT = 2
    config.MAX_RATIO_NEG_POS = np.inf#10.0
    return config

def writeDict(dict, filename):
    import datetime
    with open(filename, "a") as f:
        f.write(str(datetime.datetime.now()) + "\n")
        for i in dict.keys():            
            f.write(i + " : {}".format(dict[i]) + "\n")


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    assert os.path.exists(args.outdir), 'SVM path does not exist'
    sigmoid_cfg = get_sigmoid_configuration()
    #write configuration file to dir
    out_dir = args.outdir
    writeDict(sigmoid_cfg, '{}/sigmoid_cfg.txt'.format(out_dir))
    # Must turn this off to prevent issues when digging into the net blobs to
    # pull out features (tricky!)
    cfg.DEDUP_BOXES = 0
    # Must turn this on because we use the test im_detect() method to harvest
    # hard negatives
    cfg.TEST.SVM = True
    cfg.TRAIN.USE_FLIPPED = sigmoid_cfg.USE_FLIPPED

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    #load dataset
    imdb_name = args.imdb_name + '.' + args.image_set
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    if args.k_fold: imdb.gen_k_fold(args.k_fold)
 
    #set classes to train
    classes_to_train = range(args.first_class,min(args.first_class+args.class_num,imdb.num_classes))
    
    # enhance roidb to contain flipped examples
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'
    #train folds
    if args.k_fold:
        print 'Training with {:d} folds'.format(args.k_fold)
        for k in range(args.k_fold):
            #load trained network
            svm_file = os.path.join(out_dir,'svm_fold_{}.pkl'.format(k))
            assert os.path.exists(svm_file), 'SVM weight file does not exist'
            load_svm_coeffs(net,svm_file,classes_to_train)
            fitter = sigmoidFitter(net, imdb,sigmoid_cfg,classes_to_train,k)
            fitter.fit()
            fitter.save_sigmoids('{}/sigmoid_fold_{}_cls_{:d}_{:d}.pkl'.format(out_dir,k,classes_to_train[0],classes_to_train[-1]))
            print 'Wrote sigmoid model to: {:s}'.format('{}/sigmoid_fold_{}_cls_{:d}_{:d}.pkl'.format(out_dir,k,classes_to_train[0],classes_to_train[-1]))
    #train on full training set
    else:
        svm_file = os.path.join(out_dir,'svm.pkl')
        assert os.path.exists(svm_file), 'SVM weight file does not exist'
        load_svm_coeffs(net,svm_file,classes_to_train)
        fitter = sigmoidFitter(net, imdb,sigmoid_cfg,classes_to_train)
        fitter.fit()
        fitter.save_sigmoids('{}/sigmoid_cls_{:d}_{:d}.pkl'.format(out_dir,classes_to_train[0],classes_to_train[-1]))
        print 'Wrote sigmoid model to: {:s}'.format('{}/sigmoid_cls_{:d}_{:d}.pkl'.format(out_dir,classes_to_train[0],classes_to_train[-1]))
