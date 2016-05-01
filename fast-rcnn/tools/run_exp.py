#!/usr/bin/env python
# --------------------------------------------------------
# Run full experiment:
# Generate caffe models, run training and test on training and validation sets
# --------------------------------------------------------
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../','init_paths.py'))
import sys
from costume_net_generator.generator import gen_proto_files

#experiment params
TRAIN = True
TEST = False
training_iterations = '80000'
data_path=os.path.join(paths.DATASETS_BASE,'DisneyCharacter/')
labels_file = os.path.join(paths.DATASETS_BASE,'DisneyCharacter/labels.txt')
model_name = 'CaffeNet'
model_base_dir = os.path.join(paths.FAST_RCNN_BASE,'models')
gpu_id = '0'
finalWeightFile = os.path.join(paths.DATASETS_BASE,'DisneyCharacter/train_output/default/caffenet_fast_rcnn_iter_'+training_iterations+'.caffemodel')

#generate net according to number of classes
files = gen_proto_files(os.path.join(model_base_dir,model_name),model_base_dir,labels_file)

#training
if TRAIN:
    cmd = "./tools/train_net.py "
    cmd+= str("--gpu " + gpu_id + " ")
    cmd+= str("--iters " + training_iterations + " ")
    cmd+=str("--solver " + files['solver'] + " ")
    cmd+=str("--weights data/imagenet_models/" + model_name + ".v2.caffemodel ")
    os.system(cmd)

#testing
if TEST:
    cmd = "./tools/evaluate_net.py "
    cmd+= str("--gpu " + gpu_id + " ")
    cmd+= "--set test "
    cmd+=str("--def " + files['test'] + " ")
    cmd+=str("--net " + finalWeightFile + " ")
    os.system(cmd)
