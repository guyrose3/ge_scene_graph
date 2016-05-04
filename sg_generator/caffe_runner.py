#! /usr/bin/env python
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'..','init_paths.py'))
import numpy as np
import dlib
import cv2
from detector.obj_detector import detector
import caffe

def get_obj_proposals(image_path):
	img = cv2.imread(image_path)
	rects = []
	dlib.find_candidate_object_locations(img, rects, min_size=500,kvals=(100,100,1))
	#dlib.find_candidate_object_locations(img, rects, min_size=500), default activation
	#convert to fast-rcnn format
	boxes = np.zeros((0,4),dtype=np.float)
	for r in rects:
		boxes = np.vstack((boxes,np.array([r.left(),r.top(),r.right(),r.bottom()],dtype=np.float)))
	return boxes

class sg_caffe_runner(object):
	def __init__(self,object_path,attribute_path,gpu_mode):
		self.gpu_mode = gpu_mode
		self.object_det = {}
		self.attribute_det = {}
		self.object_det['path'] = object_path
		self.attribute_det['path'] = attribute_path
		self.load_caffe_nets()

	def load_caffe_nets(self):
		#pretrained imagenet weights
		caffemodel = os.path.join(paths.FAST_RCNN_BASE,'data/imagenet_models/CaffeNet.v2.caffemodel')
		
		#load object
		obj_class_path = os.path.join(self.object_det['path'],'labels.txt')
		obj_prototxt   = os.path.join(self.object_det['path'],'test.prototxt')
		svm_path_obj   = os.path.join(self.object_det['path'],'train_output/train.default')
		self.object_det['detector'] = detector(obj_prototxt,caffemodel,obj_class_path,gpu_mode=self.gpu_mode,\
											   use_svm=True,output_all_boxes=True,svm_path=svm_path_obj)
		#load attribute
		att_class_path = os.path.join(self.attribute_det['path'],'labels.txt')
		att_prototxt   = os.path.join(self.attribute_det['path'],'test.prototxt')
		svm_path_att   = os.path.join(self.attribute_det['path'],'train_output/train.default')
		self.attribute_det['detector'] = detector(att_prototxt,caffemodel,att_class_path,gpu_mode=self.gpu_mode,\
											      use_svm=True,output_all_boxes=True,svm_path=svm_path_att)
	
	def get_object_detections(self,image_path):
		self.run_detection(self.object_det['detector'],image_path)
		return self.object_det['detector'].detections


	def get_attribute_detections(self,image_path):
		self.run_detection(self.attribute_det['detector'],image_path)
		return self.attribute_det['detector'].detections


	def run_detection(self,detector,image_path):
		#generate object proposals
		boxes = get_obj_proposals(image_path)
		#run detector
		detector.detect_single_aux(image_path,boxes)

	def get_object_labels(self):
		return self.object_det['detector'].classes

	def get_attribute_labels(self):
		return self.attribute_det['detector'].classes
