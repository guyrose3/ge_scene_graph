# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.general_dataset
import numpy as np
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../../','init_paths.py'))

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.disney_dataset() #see what kind of parameters sould be added
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

#databases for disney characters
name = 'disney_dataset'
data_path = os.path.join(paths.DATASETS_BASE,'DisneyCharacter')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))

#databases for scene graph full
name = 'sg_dataset_objects_266'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/objects_266')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))
name = 'sg_dataset_attributes_145'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/attributes_145')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))
name = 'sg_dataset_objects_attributes_2295'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_2295')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))
name = 'sg_dataset_objects_attributes_984'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_984')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))
name = 'sg_dataset_objects_relationships_303'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/objects_relationships_303')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))
name = 'sg_dataset_relationships_objects_317'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/relationships_objects_317')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))


#databases for scene graph sanity
name = 'sg_dataset_objects'
data_path = os.path.join(paths.DATASETS_BASE,'sg_sanity/objects_final')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))

name = 'sg_dataset_attributes'
data_path = os.path.join(paths.DATASETS_BASE,'sg_sanity/attributes_final')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))

name = 'sg_dataset_objects_attributes'
data_path = os.path.join(paths.DATASETS_BASE,'sg_sanity/objects_attributes_final')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))


#databases for scene graph relationship sanity
name = 'sg_dataset_objects_relationships'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/objects_relationships')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))

name = 'sg_dataset_relationships_objects'
data_path = os.path.join(paths.DATASETS_BASE,'sg_full/relationships_objects')
for image_set in ['train','test']:
    __sets[name +'.'+image_set] = (lambda name=name,image_set=image_set, data_path=data_path: datasets.general_dataset(name,image_set, data_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
