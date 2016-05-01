#!/usr/bin/env python

# Written by Guy Rosenthal
'''
Genenrate net weights for a subset net out of a bigger net
'''
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../','init_paths.py'))
import shutil
import numpy as np
import cPickle as pickle

def generate_mapping(src_dir,dst_dir):
    src_file = os.path.join(src_dir,'../../labels.txt')
    dst_file = os.path.join(dst_dir,'../../labels.txt')
    with open(src_file,'r') as f_in : src_labels = ['__background__'] + f_in.readlines()
    with open(dst_file,'r') as f_out: dst_labels = ['__background__'] + f_out.readlines()
    mapping = [0] * len(dst_labels)
    for i in range(len(mapping)):
        mapping[i] = src_labels.index(dst_labels[i])
    return mapping

def generate_svm(src_dir,mapping):
    svm_file = os.path.join(src_dir,'svm.pkl')
    src_w,src_b = pickle.load(open(svm_file,'rb'))
    dst_w = np.zeros((len(mapping),src_w.shape[1]))
    dst_b = np.zeros((len(mapping)))
    for i in range(len(mapping)):
        dst_w[i,:]  = src_w[mapping[i],:]
        dst_b[i]    = src_b[mapping[i]]
    return [dst_w,dst_b] 

def generate_sigmoid(src_dir,mapping):
    sigmoid_file = os.path.join(src_dir,'sigmoid.pkl')
    src_sigmoid = pickle.load(open(sigmoid_file,'rb'))
    dst_sigmoid = []
    for i in range(len(mapping)):
        dst_sigmoid.append(src_sigmoid[mapping[i]])
    return dst_sigmoid

    

src_net_dir = os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_2295/train_output/train.default')
dst_net_dir = os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_984/train_output/train.default')

assert os.path.exists(src_net_dir), 'Source path for net does not exist: {}'.format(src_net_dir)
if not os.path.exists(dst_net_dir): os.makedirs(dst_net_dir)
#generate mapping between 2 nets
class_map = generate_mapping(src_net_dir,dst_net_dir)
#generate svm weight file
svm_file  = generate_svm(src_net_dir,class_map)
#generate sigmoid weight file
sigmoid_file  = generate_sigmoid(src_net_dir,class_map)
#save files to dst directory
pickle.dump(svm_file    ,open(os.path.join(dst_net_dir,'svm.pkl'    ),'wb'))
pickle.dump(sigmoid_file,open(os.path.join(dst_net_dir,'sigmoid.pkl'),'wb'))
shutil.copy(os.path.join(src_net_dir,'sigmoid_cfg.txt'),dst_net_dir)
shutil.copy(os.path.join(src_net_dir,'svm_cfg.txt'    ),dst_net_dir)



