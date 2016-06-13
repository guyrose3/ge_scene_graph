#! /usr/bin/env python
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'..','init_paths.py'))
import numpy as np
import dlib
import cv2
import pickle
from detector.obj_detector import detector
import caffe
from general_obj_att_rel import general_obj_att_rel

#############################################
#Helper Functions
#############################################
def get_obj_proposals(image_path):
	img = cv2.imread(image_path)
	rects = []
	dlib.find_candidate_object_locations(img, rects, min_size=500,kvals=(100,100,1))
	#dlib.find_candidate_object_locations(img, rects, min_size=500) #default activation
	#convert to fast-rcnn format
	boxes = np.zeros((0,4),dtype=np.float)
	for r in rects:
		boxes = np.vstack((boxes,np.array([r.left(),r.top(),r.right(),r.bottom()],dtype=np.float)))
	return boxes

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


#############################################
#Object/Attribute
#############################################
class v1_object(object):
	def __init__(self,data_path=os.path.join(paths.CAFFE_DETECTORS_BASE,'object'),gpu_mode=False):
		self.gpu_mode = gpu_mode
		#pretrained imagenet weights
		caffemodel = os.path.join(paths.FAST_RCNN_BASE,'data/imagenet_models/CaffeNet.v2.caffemodel')
		#load detector
		obj_class_path = os.path.join(data_path,'labels.txt')
		obj_prototxt   = os.path.join(data_path,'test.prototxt')
		svm_path_obj   = os.path.join(data_path,'train_output/train.default')
		self.detector = detector(obj_prototxt,caffemodel,obj_class_path,gpu_mode=self.gpu_mode,\
								 use_svm=True,output_all_boxes=True,svm_path=svm_path_obj)

	def class_names(self):
		return self.detector.classes

	def run_model(self,images,boxes=None):
		detections = []
		if boxes is None: out_boxes = []
		for i,image_path in enumerate(images):
			if boxes is not None:
				im_boxes = boxes[i]
			else:
				im_boxes = get_obj_proposals(image_path)
			#run detector
			self.detector.detect_single_aux(image_path, im_boxes)
			detections.append(self.detector.detections)
			if boxes is None: out_boxes.append(im_boxes)

		if boxes==None: return (detections,out_boxes)
		else:			return detections

#############################################
#Relationship
#############################################
class v1_relationship(object):
	def __init__(self,rel_models_file=os.path.join(paths.PROJECT_ROOT,'data/scene_graph_model_o1_r_o2.pkl'),
					  rel_prior=None,top_k=3,\
					  rel_labels_file = os.path.join(paths.PROJECT_ROOT,'labels/rel_labels.txt')):
		self.rel_models = pickle.load(open(rel_models_file, 'rb'))
		self.obj_list   = []
		self.rel_prior = rel_prior
		self.rel_list = []
		self.top_k = top_k
		self.rel_names = self.get_class_names(rel_labels_file)

	def get_class_names(self,labels_file):
		with open(labels_file) as f:
			names = f.readlines()
			names = [n.strip() for n in names]
		return names

	def class_names(self):
		return self.rel_names

	def run_model(self,boxes,rel_idx,box_obj_idx):
		rel_list = []
		if rel_idx is None: rel_idx = list(range(len(self.rel_names)))
		if box_obj_idx is None: box_obj_idx = [-1] * len(boxes)
		for idx1 in range(len(boxes)):
			obj1 = {'bbox' : boxes[idx1], 'object' : box_obj_idx[idx1]}
			for idx2 in range(idx1 + 1, len(boxes)):
				obj2 = {'bbox': boxes[idx2], 'object': box_obj_idx[idx2]}
				pair_rels = self._run_pair(obj1, obj2,rel_idx)
				rel_list.append({'o1': idx1, 'o2': idx2, 'rels': pair_rels})

		return rel_list


	def _run_pair(self,obj1,obj2,rel_idx):
		rel_probs = {'o1_o2' : np.zeros(len(rel_idx)),'o2_o1' :np.zeros(len(rel_idx))}
		for idx,rel_num in enumerate(rel_idx):
			rel_models = self.rel_models[rel_num]
			model = self.get_model(obj1['object'],obj2['object'],rel_models)
			rel_probs['o1_o2'][idx] = calc_bb_probability(model,obj1['bbox'],obj2['bbox'])
			model = self.get_model(obj2['object'],obj1['object'],rel_models)
			rel_probs['o2_o1'][idx] = calc_bb_probability(model,obj2['bbox'],obj1['bbox'])

		'''TODO-remove later
		#keep top k for each side
		top_k_1 = np.argsort(-rel_probs['o1_o2'])[:self.top_k]
		rel_probs['o1_o2'] = np.vstack((top_k_1,rel_probs['o1_o2'][top_k_1]))
		top_k_2 = np.argsort(-rel_probs['o2_o1'])[:self.top_k]
		rel_probs['o2_o1'] = np.vstack((top_k_2,rel_probs['o2_o1'][top_k_2]))
		'''

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
#############################################
#Main Module
#############################################
class grader(general_obj_att_rel):
	'''
	This class implements general api to object,attribute and relationship detectors
	It extracts information from relevant modules
	'''
	def __init__(self):
		self._object = v1_object()
		self._attribute = v1_object(data_path=os.path.join(paths.CAFFE_DETECTORS_BASE,'attribute'))
		self._relationship = v1_relationship()

	
	

