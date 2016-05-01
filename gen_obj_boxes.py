#!/usr/bin/env python
import json
import numpy as np
import pickle
import pdb

def convert_bbox(bb):
    return np.array([max(0,bb['x']),max(0,bb['y']),bb['x']+bb['w'],bb['y']+bb['h']],dtype=np.uint16)

#load dataset annotations
with open("sg_test_annotations_clean.json") as f: data_in = json.load(f)

#extract gt for each object
data_out = []
for d in data_in:
    image_reg = {}
    image_reg['filename'] = d['filename']
    image_reg['objects'] = []
    for o in d['objects']:
        single_reg = {}
        single_reg['object'] = o['names'][0]
        single_reg['bbox']   = convert_bbox(o['bbox'])
        image_reg['objects'].append(single_reg)
    data_out.append(image_reg)

#store object annotations
with open("sg_test_obj_bbox.pkl",'w') as f: pickle.dump(data_out,f)

pdb.set_trace()
