#!/usr/bin/env python
# --------------------------------------------------------
# Run full experiment:
# Generate caffe models, run training and test on training and validation sets
# --------------------------------------------------------
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../','init_paths.py'))
import sys
import time
from costume_net_generator.generator import gen_proto_files

#experiment params
TRAIN_SVM = True
TEST = False
k_fold = '8'

#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_266')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/attributes_145')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_2295')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_relationships_303')
data_path=os.path.join(paths.DATASETS_BASE,'sg_full/relationships_objects_317')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_984')

labels_file = os.path.join(data_path,'labels.txt')

#imdb_name = 'sg_dataset_objects_266'
#imdb_name = 'sg_dataset_attributes_145'
#imdb_name = 'sg_dataset_objects_attributes_2295'
#imdb_name = 'sg_dataset_objects_relationships_303'
imdb_name = 'sg_dataset_relationships_objects_317'
#imdb_name = 'sg_dataset_objects_attributes_984'

model_name = 'CaffeNet'
model_base_dir = os.path.join(paths.FAST_RCNN_BASE,'models')
gpu_id = '2'
weight_file_name = str(model_name + ".v2_svm.caffemodel")
finalWeightFile = data_path + '/train_output/default/' + weight_file_name
if k_fold: out_dir = os.path.join(data_path,'train_output','train.k_fold.' + time.strftime("%Y%m%d-%H%M%S"))
else     : out_dir = os.path.join(data_path,'train_output','train.' + time.strftime("%Y%m%d-%H%M%S"))

#generate net according to number of classes
files = gen_proto_files(os.path.join(model_base_dir,model_name),model_base_dir,labels_file)

#training
if TRAIN_SVM:
    cmd = "./tools/train_svms.py "
    cmd+= str("--gpu " + gpu_id + " ")
    cmd+=str("--def " + files['test'] + " ")
    cmd+=str("--net data/imagenet_models/" + model_name + ".v2.caffemodel ")
    cmd+= str("--imdb " + imdb_name + " ")
    cmd+=str("--outdir " + out_dir + " ")
    if k_fold:
        cmd+=str("--k_fold " + k_fold + " ")
    os.system(cmd)

#testing
if TEST:
    cmd = "./tools/evaluate_net.py "
    cmd+= str("--gpu " + gpu_id + " ")
    cmd+= str("--imdb " + imdb_name + " ")
    cmd+= "--set test "
    cmd+=str("--def " + files['test'] + " ")
    cmd+=str("--net " + finalWeightFile + " ")
    cmd+=str("--datapath " + data_path + " ")
    cmd+="--conf 0.3 "
    cmd+= "--svm False "
    os.system(cmd)
