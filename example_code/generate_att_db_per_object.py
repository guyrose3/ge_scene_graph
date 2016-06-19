#! /usr/bin/env python

'''
This script generates object database features, along with attribute annotations
features are extracted from faster r-cnn net and will be used for attributes prediction given the object
'''
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../','init_paths.py'))
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
import cPickle as pickle
import json
import numpy as np
import caffe
import cv2
import pdb

################################
#Defines
################################
images_path = os.path.join(paths.DATASETS_BASE,'vg_dataset','images')
feat_layer = 'fc7'
cfg.TEST.HAS_RPN = False
cfg.DEDUP_BOXES = 0
################################
#Helper Functions
################################
def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]
    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"
    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),return_index=return_index,return_inverse=return_inverse)
    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

def eq_boxes(bb1,bb2):
    return np.abs(bb1-bb2).sum()<1

def convert_bbox(bb):
    #TODO-see if +/-1 is needed in the conversion
    #TODO-add object specific attributes
    return np.array([max(0,bb['x']),max(0,bb['y']),bb['x']+bb['w'],bb['y']+bb['h']],dtype=np.float)

def get_obj_att_names():
    coco_obj_file = 'coco_obj.txt'
    with open(coco_obj_file) as f: obj_names = f.readlines()
    obj_names = [n.strip() for n in obj_names]
    ret_val = []
    for o in obj_names:
        ret_val.append({'obj_name': o, 'attributes' : []})
    return ret_val

def alloc_example_space(att_data,names,feature_dim=4096):
    print 'Allocating space for positive examples'
    #object names
    obj = [l['obj_name'] for l in names]
    num_pos = np.zeros(len(obj),dtype=np.int)
    for im_idx,d in enumerate(att_data):
        print 'Scanning image {}/{}'.format(im_idx+1,len(att_data))
        for reg in d['attributes']:
            if len(reg['object_names'])!=1: continue
            if reg['object_names'][0] in obj:
                num_pos[obj.index(reg['object_names'][0])]+=1
    #generate database
    print 'Generating database'
    pos_db = []
    for idx in range(len(num_pos)):
        pos_db.append(np.zeros((num_pos[idx],feature_dim),dtype=np.float))

    return (obj,pos_db)

def load_coco_net(gpu_id=0):
    caffemodel = os.path.join(paths.DATASETS_BASE,'faster_rcnn_data/data/faster_rcnn_models/coco_vgg16_faster_rcnn_final.caffemodel')
    #prototxt   = os.path.join(paths.FAST_RCNN_BASE,'models/coco/VGG16/faster_rcnn_end2end/test.prototxt')
    prototxt   = os.path.join(paths.FAST_RCNN_BASE,'models/coco/VGG16/fast_rcnn/test.prototxt')
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net


def get_box_attributes(boxes,obj_idx,d,obj_att_data):
    '''Exrract attributes accosiated with each specific bounding box and its object'''
    all_atts = [0] * len(boxes)
    for box_idx,box in enumerate(boxes):
        box_atts = []
        obj_cls_idx = obj_idx[box_idx]
        for reg in d['attributes']:
            if not eq_boxes(box,convert_bbox(reg)): continue
            if not obj_att_data[obj_cls_idx]['obj_name']==reg['object_names'][0]: continue
            for single_att in reg['attributes']:
                if single_att in obj_att_data[obj_cls_idx]['attributes']:
                    box_atts.append(single_att)
        #assign with no repetitions
        all_atts[box_idx] = list(set(box_atts))

    return all_atts


def extract_positives(net,pos,att_data,obj_names):
    #load attributes per object
    obj_att_data = pickle.load(open('atts_per_obj_v1.pkl','rb'))
    att_output = [[] for _ in range(len(obj_names))]
    #define idx counter
    curr_idx = np.zeros(len(obj_names),dtype=np.int)
    for im_idx,d in enumerate(att_data):
        print 'Extracting features from image {}/{}'.format(im_idx+1,len(att_data))
        boxes = np.zeros((0,4),dtype=np.float)
        obj_idx = []
        for reg in d['attributes']:
            if len(reg['object_names'])!=1: continue
            if reg['object_names'][0] in obj_names:
                obj_idx.append(obj_names.index(reg['object_names'][0]))
                boxes = np.vstack((boxes,convert_bbox(reg)))

        if boxes.shape[0]==0:continue
        #get unique boxes to run
        obj_idx = np.asarray(obj_idx,dtype=np.int)
        boxes,unique_idx = unique_rows(boxes, return_index=True)
        obj_idx = obj_idx[unique_idx]

        #get attributes associated to each bounding box
        image_atts = get_box_attributes(boxes,obj_idx,d,obj_att_data)

        for i,cls_idx in enumerate(obj_idx):
            att_output[cls_idx].append(image_atts[i])

        #propagate boxes through neural network
        im = cv2.imread(os.path.join(images_path,'{}.jpg'.format(d['id'])))
        #TODO-handle flipped examples??
        _, _ = im_detect(net, im, boxes)
        feat = net.blobs[feat_layer].data
        assert feat.shape[0]==boxes.shape[0], '{} boxes yields {} features'.format(boxes.shape[0],feat.shape[0])
        #populate positive database
        for pos_idx,ex in enumerate(feat):
            class_idx = obj_idx[pos_idx]
            pos[class_idx][curr_idx[class_idx]] = ex
            curr_idx[class_idx]+=1

        #verify even features and attributes numbers
        att_num_ver = np.array([len(a) for a in att_output],dtype=np.int)
        assert np.all(att_num_ver==curr_idx), 'Wrong assignment in assignment\n att = {}\n feat = {}'.format(att_num_ver,curr_idx)

    return att_output

#example line of the database TODO-remove later
#att_data[0]['attributes'][0]
#{u'h': 339, u'object_names': [u'clock'], u'w': 79, u'x': 421, u'y': 91, u'attributes': [u'green'], u'id': 38203}


#load network
net = load_coco_net()

#open attribute files
with open('attributes.json') as f: att_data = json.load(f)

#get object names
names =  get_obj_att_names()

#allocate space for positive examples
obj_names,pos = alloc_example_space(att_data,names)

#get examples, with their corresponding attributes
att_data = extract_positives(net,pos,att_data,obj_names)

#save results
feat_out_file = os.path.join(paths.DATASETS_BASE,'vg_out_coco_categories.pkl')
pickle.dump(pos,open(feat_out_file,'wb'))
att_out_file = os.path.join(paths.DATASETS_BASE,'vg_out_coco_categories_attributes.pkl')
pickle.dump(att_data,open(att_out_file,'wb'))
