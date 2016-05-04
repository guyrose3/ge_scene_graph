#! /usr/bin/env python
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'..','init_paths.py'))
import numpy as np

def calc_bbs_features(bb1,bb2):
    bb1 = bb1.astype(np.float)
    bb2 = bb2.astype(np.float)
    f1 = (bb1[0]-bb2[0])/(bb1[2]-bb1[0])
    f2 = (bb1[1]-bb2[1])/(bb1[3]-bb1[1])
    f3 = (bb1[2]-bb1[0])/(bb2[2]-bb2[0])
    f4 = (bb1[3]-bb1[1])/(bb2[3]-bb2[1])
    return np.array([f1,f2,f3,f4])
    
def calc_bb_probability(model,bb1,bb2):
    #calc features
    if 'reverse' in model.keys() and model['reverse']: f = calc_bbs_features(bb2,bb1)
    else                                             : f = calc_bbs_features(bb1,bb2)
    #apply gmm
    dim = f.shape[0]
    density = 0.0
    for i in range(model['weights'].shape[0]):
        mu = model['means'][i]
        w  = model['weights'][i]
        if len(model['covars'][i].shape)==2:
            covar = model['covars'][i]
        else:
            covar = np.diag(model['covars'][i])
        f_c = f-mu
        inv_cov = np.linalg.inv(covar)
        nom = np.exp(-0.5 * np.dot(f_c.T , np.dot(inv_cov , f_c)))
        denom = np.sqrt(((2*np.pi)**dim) * np.linalg.det(covar))
        density = density + (w*(nom/denom))
    #apply sigmoid to obtain probability
    return 1. / (1. + np.exp(density * model['sigmoid'][0] + model['sigmoid'][1]))

###########################
#relationship runner module
###########################
class rel_runner(object):
    def __init__(self,rel_models,rel_prior=None,top_k=3,\
				 rel_labels_file = os.path.join(paths.PROJECT_ROOT,'labels/rel_labels.txt')):
        self.rel_models = rel_models
        self.obj_list   = []
        self.rel_prior = rel_prior
        self.rel_list = []
        self.top_k = top_k

    def set_image_objects(self,obj_list):
        self.obj_list = obj_list

    def get_rel_list(self):
        rel_list = self.rel_list
        self.rel_list = []
        return rel_list

    def run(self):
        '''iterate all object pairs with all possible relationships'''
        for idx1 in range(len(self.obj_list)):
            obj1 = self.obj_list[idx1]
            for idx2 in range(idx1+1,len(self.obj_list)):
                obj2 = self.obj_list[idx2]
                pair_rels = self._run_pair(obj1,obj2)
                self.rel_list.append({'o1':idx1,'o2':idx2,'rels':pair_rels})


    def _run_pair(self,obj1,obj2):
        rel_probs = {'o1_o2' : np.zeros(len(self.rel_models)),'o2_o1' :np.zeros(len(self.rel_models))}
        for idx,rel_models in enumerate(self.rel_models):
            model = self.get_model(obj1['object'],obj2['object'],rel_models)
            rel_probs['o1_o2'][idx] = calc_bb_probability(model,obj1['bbox'],obj2['bbox'])
            model = self.get_model(obj2['object'],obj1['object'],rel_models)
            rel_probs['o2_o1'][idx] = calc_bb_probability(model,obj2['bbox'],obj1['bbox'])
        #keep top k for each side
        top_k_1 = np.argsort(-rel_probs['o1_o2'])[:self.top_k]
        rel_probs['o1_o2'] = np.vstack((top_k_1,rel_probs['o1_o2'][top_k_1]))
        top_k_2 = np.argsort(-rel_probs['o2_o1'])[:self.top_k]
        rel_probs['o2_o1'] = np.vstack((top_k_2,rel_probs['o2_o1'][top_k_2]))

        return rel_probs

    def get_model(self,obj1_idx,obj2_idx,rel_models):
        selected_sub_model = rel_models[0]#object agnostic model
        for sub_model in rel_models:
            if tuple(sub_model['objects'])==(obj1_idx,obj2_idx):
                selected_sub_model = sub_model
                break
        #add relationship to tables
        model = {'weights'  : selected_sub_model['weights'],
                 'means'    : selected_sub_model['means'],
                 'covars'   : selected_sub_model['covars'],
                 'sigmoid'  : selected_sub_model['sigmoid']}         

        return model

