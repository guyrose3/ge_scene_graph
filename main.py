#! /usr/bin/env python
import cPickle as pickle
from utils import convert_obj_data,rel_runner,scene_viewer
import numpy as np

obj_att_data_file = 'data/obj_att_data.pkl'
rel_models_file   = 'data/scene_graph_model_o1_r_o2.pkl'
obj_labels_file   = 'labels/obj_labels.txt'
att_labels_file   = 'labels/att_labels.txt'
rel_labels_file   = 'labels/rel_labels.txt'

#parameters
num_top_k = 3

#defines
images_dir = 'images'
RANK_RELATIONSHIPS = False
RANK_ATTRIBUTES    = False
VIEW_SCENE_OUTPUT  = True

if __name__=='__main__':
	print 'Start Scene Graph generator'
	if RANK_RELATIONSHIPS:
		print 'Generating top k relationship scores'
		#load objects + attributes scores
		data = pickle.load(open(obj_att_data_file,'rb'))
		#convert object data to object indices
		obj_data = convert_obj_data(data,obj_labels_file)
		#load relationship models
		rel_models = pickle.load(open(rel_models_file,'rb'))
		#define relationship runner
		rel = rel_runner(rel_models,top_k=num_top_k)
		print 'Start iterating over image data' 
		#iterate over images
		for image_idx in range(len(obj_data)):
			print 'Running relationships over  image {:d}/{:d}'.format(image_idx+1,len(obj_data))
			rel.set_image_objects(obj_data[image_idx]['objects'])
			rel.run()
			rel_list = rel.get_rel_list()
			data[image_idx]['relationships'] = rel_list

		pickle.dump(data,open('data/obj_att_rel_data.pkl','wb'))

	if RANK_ATTRIBUTES:
	   	print 'Generating top k attribute scores'
		#load objects + attributes scores
   		data = pickle.load(open('data/obj_att_rel_data.pkl','rb'))
		for im_idx in range(len(data)):
			for obj_idx in range(len(data[im_idx]['objects'])):
				top_k = np.argsort(-data[im_idx]['objects'][obj_idx]['attributes_prob'])[:num_top_k]
				data[im_idx]['objects'][obj_idx]['attributes_prob'] = \
				np.vstack((top_k,data[im_idx]['objects'][obj_idx]['attributes_prob'][top_k]))
	
		import pdb;pdb.set_trace()
		pickle.dump(data,open('data/obj_att_rel_data_top_{}_only.pkl'.format(num_top_k),'wb'))

	if VIEW_SCENE_OUTPUT:
		data = pickle.load(open('data/obj_att_rel_data_top_{}_only.pkl'.format(num_top_k),'rb'))
		viewer = scene_viewer(att_labels_file,rel_labels_file,images_dir)
		for im_idx,d in enumerate(data):
			print 'showing scenes for image {:d}/{:d}'.format(im_idx+1,len(data))
			viewer.set_image_data(d)
			viewer.view_attributes()
			viewer.view_relationships()
			#debug
			import pdb;pdb.set_trace()


		

