#!/usr/bin/env python
# --------------------------------------------------------
# Run full experiment:
# Generate caffe models, run training and test on training and validation sets
# --------------------------------------------------------
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../','init_paths.py'))
import sys
import time
from costume_net_generator.generator import gen_proto_files,gen_proto_files_by_num_classes
import subprocess

def merge_sigmoid_params(output_dir,num_classes,k_fold_idx=None):
    import numpy as np
    import cPickle as pickle
    #get list of all files
    all_files = [f for f in os.listdir(output_dir) if os.path.splitext(f)[1]=='.pkl']
    if k_fold_idx is not None: num_parts = 6
    else:                      num_parts = 4
    #################pack sigmoid files######################################
    sigmoid_pack = [[] for _ in range(num_classes)]
    for f in all_files:
        parts = (os.path.splitext(f)[0]).split('_')
        if (not parts[0]=='sigmoid') or len(parts)!=num_parts: continue
        if k_fold_idx is not None and int(parts[2])!=k_fold_idx:continue
        sig_param = pickle.load(open(os.path.join(output_dir,f), 'rb'))
        idx_s = int(parts[-2]);idx_f = int(parts[-1]) 
        sigmoid_pack[idx_s:idx_f+1] = sig_param
    if k_fold_idx is not None: file_name = 'sigmoid_fold_{}.pkl'.format(k_fold_idx)
    else:             file_name = 'sigmoid.pkl'
    pickle.dump(sigmoid_pack,open(os.path.join(output_dir,file_name), 'wb'))



#experiment params
TRAIN_SVM = True
TEST = False
k_fold = None#'3'

#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_266')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/attributes_145')
data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_attributes_2295')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/objects_relationships')
#data_path=os.path.join(paths.DATASETS_BASE,'sg_full/relationships_objects')

labels_file = os.path.join(data_path,'labels.txt')

#imdb_name = 'sg_dataset_objects_266'
#imdb_name = 'sg_dataset_attributes_145'
imdb_name = 'sg_dataset_objects_attributes_2295'
#imdb_name = 'sg_dataset_objects_relationships_303'
#imdb_name = 'sg_dataset_relationships_objects_317'

model_name = 'CaffeNet'
model_base_dir = os.path.join(paths.FAST_RCNN_BASE,'models')
weight_file_name = str(model_name + ".v2_svm.caffemodel")
finalWeightFile = data_path + '/train_output/default/' + weight_file_name
#TODO-choose existing dir
out_dir = os.path.join(data_path,'train_output','train.20151208-085223')

#generate net according to number of classes
files = gen_proto_files(os.path.join(model_base_dir,model_name),model_base_dir,labels_file)

#training-run svm in parallel and merge results in the end
if TRAIN_SVM:
    #train dimensions
    gpus = [0,1,2,3,4,5,6,7]
    num_classes_per_process = 8

    #generate the net with number of needed classes
    files = gen_proto_files_by_num_classes(os.path.join(model_base_dir,model_name),model_base_dir,num_classes_per_process)

    
    with open(labels_file,'r') as f: last_class=len(f.readlines())
    train_needed = range(1,last_class+1)

    cls = train_needed[0]
    tmp_idx = 0

    used_gpus = 0
    proc = []
    while cls<=last_class:
        cmd = []
        cmd.append("python")
        cmd.append("tools/train_sigmoids_parallel.py")
        cmd.append("--gpu")        ; cmd.append(str(gpus[used_gpus]))
        cmd.append("--def")        ; cmd.append(files['test'])
        cmd.append("--net")        ; cmd.append("data/imagenet_models/" + model_name + ".v2.caffemodel")
        cmd.append("--imdb")       ;cmd.append(imdb_name)
        cmd.append("--outdir")     ;cmd.append(out_dir)
        cmd.append("--first_class");cmd.append(str(cls))
        cmd.append("--class_num")  ;cmd.append(str(num_classes_per_process))
        if k_fold:
            cmd.append("--k_fold")  ;cmd.append(k_fold)
        #run command
        print 'Strat training calasses {}-{}'.format(cls,cls+num_classes_per_process-1)
        #debug - run 1 process to see if working
        #os.system(' '.join(cmd))
        proc.append(subprocess.Popen(cmd))#don't wait till finished
        
        #update counters
        tmp_idx+=num_classes_per_process
        if tmp_idx>=len(train_needed): cls = last_class+1
        else:                          cls = train_needed[tmp_idx]


        used_gpus+=1
        if used_gpus==len(gpus) or cls>=last_class:
            exit_code = [p.wait() for p in proc]
            print 'Done training until class {}'.format(cls-1)
            proc = []
            used_gpus=0

    merge_sigmoid_params(out_dir,last_class+1)
    import pdb; pdb.set_trace()
