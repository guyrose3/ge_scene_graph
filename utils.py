#!/usr/bin/env python
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'init_paths.py'))
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import dlib
import cv2
from detector.obj_detector import detector

###########################
#helper functions
###########################
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

def convert_obj_data(data,obj_labels_file):
    with open(obj_labels_file) as f: obj_names = f.readlines()
    obj_names = ['__background__'] + [o.strip() for o in obj_names]
    data_out = []
    for d in data:
        image_reg = {}
        image_reg['objects'] = []
        image_reg['filename'] = d['filename']
        for objects in d['objects']:
            obj_reg = {}
            obj_reg['object'] = obj_names.index(objects['object'])
            obj_reg['bbox']   = objects['bbox']
            image_reg['objects'].append(obj_reg)
        data_out.append(image_reg)
    return data_out


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
    def __init__(self,rel_models,rel_prior=None,top_k=3):
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

###########################
#Scene viewer module
###########################
class scene_viewer(object):
	def __init__(self,att_label_file,rel_label_file,images_dir):
		self.image_counter = 0
		self.images_dir = images_dir
		self.results_dir = os.path.join(images_dir,'..','results')
		self.im_data = []
		#TODO-add obj labels later
		self.obj_labels = None
		with open(att_label_file,'r') as f: att_labels = f.readlines(); att_labels = ['__background__'] + [a.strip() for a in att_labels]
		self.att_labels = att_labels
		with open(rel_label_file,'r') as f: rel_labels = f.readlines(); rel_labels = [r.strip() for r in rel_labels]
		self.rel_labels = rel_labels
	
	def set_image_data(self,im_data):
		self.image_counter+=1
		self.im_data = im_data
		im_path = os.path.join(self.images_dir,self.im_data['filename'])
		if not os.path.exists(im_path):
			#try to download from hornet
			cmd  = 'sshpass -p "Pxhpkurv11" scp -r '
			cmd += 'guyrose3@hornet.drp.cs.cmu.edu:/data03/guy/sg_full/attributes_145/images/{} '.format(self.im_data['filename'])
			cmd += '{}'.format(im_path)
			os.system(cmd)

	def view_attributes(self):
		img = Image.open(os.path.join(self.images_dir,self.im_data['filename']))
		#iterate on objects and show bounding box and attributes
		for obj_idx,obj in enumerate(self.im_data['objects']):
			self._show_obj_and_att(img,obj,obj_idx)

	def view_relationships(self):
		img = Image.open(os.path.join(self.images_dir,self.im_data['filename']))
		total_rel_idx = 0
		for obj1_idx in range(len(self.im_data['objects'])):
			for obj2_idx in range(obj1_idx+1,len(self.im_data['objects'])):
				self._show_obj_and_rel(img,obj1_idx,obj2_idx,total_rel_idx)
				total_rel_idx+=1

	def _show_obj_and_att(self,img,obj,obj_idx):
		plt.cla()
		plt.imshow(img)
		#mark object
		obj_name = obj['object']#TODO-might have to change later if object is added as index
		bbox = obj['bbox']     
		plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='g', linewidth=3))
		x = (bbox[0] + bbox[2])/2
		y = bbox[1]
		s = '{}'.format(obj_name)
		plt.text(x, y, s, fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		#print attributes
		im_w,im_h = img.size
		for att_idx in range(obj['attributes_prob'].shape[1]):
			att_name  = self.att_labels[obj['attributes_prob'][0,att_idx].astype(np.int)]
			prob = obj['attributes_prob'][1,att_idx]#TODO-maybe show cofidence
			plt.text(0, 0.4*im_h + 0.1*im_h*att_idx, '{}'.format(att_name),\
 			fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		plt.title('Top atts for obj={}'.format(obj_name))
		plt.tick_params(axis='x',which='both',bottom='off',top ='off',labelbottom='off')
		plt.tick_params(axis='y',which='both',right ='off',left='off',labelleft='off')
		plt.savefig(os.path.join(self.results_dir,'im_{}_obj_att_{}'.format(self.image_counter,obj_idx)))
		#plt.show()
		plt.close()

	def _show_obj_and_rel(self,img,obj1_idx,obj2_idx,total_rel_idx):
		#find relationship
		rel = [r['rels'] for r in self.im_data['relationships'] if r['o1']==obj1_idx and r['o2']==obj2_idx][0]
		plt.cla()
		plt.imshow(img)
		#mark object
		obj1_name = self.im_data['objects'][obj1_idx]['object']#TODO-might have to change later if object is added as index
		obj2_name = self.im_data['objects'][obj2_idx]['object']#TODO-might have to change later if object is added as index
		bbox1 = self.im_data['objects'][obj1_idx]['bbox']
		bbox2 = self.im_data['objects'][obj2_idx]['bbox']     
		plt.gca().add_patch(plt.Rectangle((bbox1[0], bbox1[1]),bbox1[2] - bbox1[0],bbox1[3] - bbox1[1], fill=False,edgecolor='g', linewidth=3))
		plt.gca().add_patch(plt.Rectangle((bbox2[0], bbox2[1]),bbox2[2] - bbox2[0],bbox2[3] - bbox2[1], fill=False,edgecolor='r', linewidth=3))
		x = (bbox1[0] + bbox1[2])/2;y = bbox1[1];s = '{}'.format(obj1_name)
		plt.text(x, y, s, fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		x = (bbox2[0] + bbox2[2])/2;y = bbox2[1];s = '{}'.format(obj2_name)
		plt.text(x, y, s, fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		#print relationships
		im_w,im_h = img.size
		#o1->o2
		plt.text(0, 0.2*im_h, 'O1->O2:', fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		for rel_idx in range(rel['o1_o2'].shape[1]):
			rel_name  = self.rel_labels[rel['o1_o2'][0,rel_idx].astype(np.int)]
			prob = rel['o1_o2'][1,rel_idx]#TODO-maybe show cofidence
			plt.text(0, 0.4*im_h + 0.1*im_h*rel_idx, '{}'.format(rel_name),\
 			fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		#o2->o1
		plt.text(im_w, 0.2*im_h, 'O2->O1:', fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		for rel_idx in range(rel['o2_o1'].shape[1]):
			rel_name  = self.rel_labels[rel['o2_o1'][0,rel_idx].astype(np.int)]
			prob = rel['o2_o1'][1,rel_idx]#TODO-maybe show cofidence
			plt.text(im_w, 0.4*im_h + 0.1*im_h*rel_idx, '{}'.format(rel_name),\
 			fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
		plt.title('Top rels for O1={},O2={}'.format(obj1_name,obj2_name))
		plt.tick_params(axis='x',which='both',bottom='off',top ='off',labelbottom='off')
		plt.tick_params(axis='y',which='both',right ='off',left='off',labelleft='off')
		plt.savefig(os.path.join(self.results_dir,'im_{}_obj_rel_{}'.format(self.image_counter,total_rel_idx)))
		#plt.show()
		plt.close()	
    
###############################
#object,attribute runner module
###############################
import caffe
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

#example usage
if __name__ == '__main__':

	#detectors path
	gpu_mode = False#TODO-change according to local configuration

	object_path    = os.path.join(paths.CAFFE_DETECTORS_BASE,'object')	
	attribute_path = os.path.join(paths.CAFFE_DETECTORS_BASE,'attribute')
	
	#define the detector runner
	obj_att_detector = sg_caffe_runner(object_path,attribute_path,gpu_mode)

	#define image
	image_path = 'images/2584263186_3969c8a53e_b.jpg'

	#detect objects in image
	obj_det = obj_att_detector.get_object_detections(image_path)

	#detect attributes in image
	att_det = obj_att_detector.get_attribute_detections(image_path)
	
