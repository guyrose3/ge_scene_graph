#!/usr/bin/env/ python
import os
import cPickle as pickle
import numpy as np
from sklearn import svm
'''
This script shows how to train 1-vs-all svm for each object attribute.
'''

#define svm
params = dict()
#TODO-should extract norm to exelerate svm training
params['scale'] = 1
params['C'] = 1
params['B'] = 1
svm_trainer = svm.LinearSVC(C=params['C'], intercept_scaling=params['B'], verbose=1,
                            penalty='l2', loss='hinge',max_iter=10000, random_state=0, dual=True)

def train_single_att_svm(features,att_db,single_att):
    '''Train attribute 1-vs-all svm given the object class
    divides object featrues to positives and negatives by the existance of the attribute and runs the trainer
    '''
    pos_idx = [];neg_idx = []
    for idx,f in enumerate(features):
        if single_att in att_db[idx]:
            pos_idx.append(idx)
        else: 
            neg_idx.append(idx)

    pos = features[np.array(pos_idx,dtype=np.int)]
    neg = features[np.array(neg_idx,dtype=np.int)]

    X = np.vstack((pos, neg)) * params['scale']
    y = np.hstack((np.ones(pos.shape[0]),-np.ones(neg.shape[0])))
    svm_trainer.class_weight = {1: float(neg.shape[0]) / y.shape[0], -1: float(pos.shape[0]) / y.shape[0]}
    #train svm
    svm_trainer.fit(X, y)

    svm_data = {'w' : svm_trainer.coef_ * params['scale'], 'b' : svm_trainer.intercept_, 'attribute' : single_att}
    return svm_data


def get_obj_atts(att_db):
    possible_atts = []
    for atts in att_db:
        if len(atts)>0: possible_atts+=atts
    return list(set(possible_atts))

if __name__=='__main__':
    svms_file = '/data01/guy/tmp_svm_file.pkl'

    #load feature_file
    feature_file = '/data01/guy/vg_out_coco_categories.pkl'
    features = pickle.load(open(feature_file,'rb'))

    attribute_file = '/data01/guy/vg_out_coco_categories_attributes.pkl'
    attributes = pickle.load(open(attribute_file,'rb'))

    #run through object classes
    all_svms = [[] for _ in range(len(features))]
    for obj_idx,obj_feats in enumerate(features):
        print 'Training svms for object # {}'.format(obj_idx) 
        att_db = attributes[obj_idx]
        assert len(att_db)==obj_feats.shape[0],\
        'Non matching number of attributes to features'
        obj_atts = get_obj_atts(att_db)
        if len(obj_atts)==0: continue
        #run through all possible objects attributes 
        for att_idx,single_att in enumerate(obj_atts):
            print '    attribute {}:{}'.format(att_idx,single_att)
            all_svms[obj_idx].append(train_single_att_svm(obj_feats,att_db,single_att))

    #save svms
    pickle.dump(all_svms,open(svms_file,'wb'))








