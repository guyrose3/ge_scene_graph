#! /usr/bin/env python
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'..','init_paths.py'))
import cPickle as pickle
from relationship_runner import rel_runner
from caffe_runner import sg_caffe_runner

class scene_graph_generator(object):
	'''
	This class includes object and attribute detectors, and also relationship models
	'''
	def __init__(self):
		self.rel_runner = self.init_rel_model()
		self.obj_att_runner = self.init_obj_att_model()
		self.att_prior = None
		self.rel_prior = None
		self.params = self.get_params()
		self.image_path = []

	def init_rel_model(self,rel_models_file = os.path.join(paths.PROJECT_ROOT,'data/scene_graph_model_o1_r_o2.pkl')):
		rel_models = pickle.load(open(rel_models_file,'rb'))
		#TODO-add labels to rel_runner module
		return rel_runner(rel_models,top_k=num_top_k)

	def init_obj_att_model(self,gpu_mode = False):
		object_path    = os.path.join(paths.CAFFE_DETECTORS_BASE,'object')	
		attribute_path = os.path.join(paths.CAFFE_DETECTORS_BASE,'attribute')
		return sg_caffe_runner(object_path,attribute_path,gpu_mode)

	def get_params(self):
		#TODO-implement
		return None

	def set_image_path(self,image_path):
		self.image_path = image_path

	def create_scene_graph(self):
		'''
		This is the main function to create a scene graph from an image
		'''
		return NotImplemented





