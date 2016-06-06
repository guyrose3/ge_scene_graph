#! /usr/bin/env python
import pdb
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'..','init_paths.py'))
import v1_obj_att_rel

grader = v1_obj_att_rel.grader()

image_path = '../images/2584263186_3969c8a53e_b.jpg'

'''test operations on grader'''
#object
obj_names = grader.object_names
detections,boxes = grader.get_object_detector_data([image_path])

#attribute
#att_names = grader.attribute_names
#detections = grader.get_attribute_detector_data([image_path],boxes)

#relationship
rel_names = grader.relationship_names
rel_boxes = boxes[0][:10]
rel_scores = grader.get_relationship_data(rel_boxes)






