# --------------------------------------------------------
# Written by Guy Rosenthal
# --------------------------------------------------------

"""Generate Fast-Rcnn network of custom size"""

import caffe
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2

def change_files(templates,num_classes,output_dir):
    #set names
    train_param = caffe_pb2.NetParameter()
    with open(templates['train'], 'rt') as f:
             pb2.text_format.Merge(f.read(), train_param)
    dirname = os.path.join(output_dir,'costumized',train_param.name + '_' + str(num_classes) + '_classes')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filenames = {'solver' : str(dirname + '/solver.prototxt'), 
                 'train'  : str(dirname + '/train.prototxt') ,
                 'test'   : str(dirname + '/test.prototxt')  }
    #train
    num_layers = len(train_param.layer)
    for l in range(num_layers):
        layer = train_param.layer[l]
        if layer.name=='data':
            layer.python_param.param_str = "\'num_classes\': "  + str(num_classes)
        elif layer.name=='cls_score':
            layer.inner_product_param.num_output = num_classes
        elif layer.name=='bbox_pred':
            layer.inner_product_param.num_output = num_classes * 4
    #test
    test_param = caffe_pb2.NetParameter()
    with open(templates['test'], 'rt') as f:
             pb2.text_format.Merge(f.read(), test_param)
    num_layers = len(test_param.layer)
    for l in range(num_layers):
        layer = test_param.layer[l]
        if layer.name=='cls_score':
            layer.inner_product_param.num_output = num_classes
        elif layer.name=='bbox_pred':
            layer.inner_product_param.num_output = num_classes * 4
    #solver
    solver_param = caffe_pb2.SolverParameter()
    with open(templates['solver'], 'rt') as f:
             pb2.text_format.Merge(f.read(), solver_param)
    solver_param.train_net = filenames['train']
    #write modified files
    with open(filenames['train'] ,'w') as f:
        f.write(pb2.text_format.MessageToString(train_param))
    with open(filenames['test'] ,'w') as f:
        f.write(pb2.text_format.MessageToString(test_param))
    with open(filenames['solver'] ,'w') as f:
        f.write(pb2.text_format.MessageToString(solver_param))
    
    return filenames

    

def gen_proto_files(input_dir,output_dir,labels_file):
    #get number of classes
    with open(labels_file) as f:
        num_classes = sum(1 for _ in f) + 1#add one for background class
    templates = {'solver' : str(os.path.join(input_dir,'solver.prototxt')),
                 'train'  : str(os.path.join(input_dir,'train.prototxt')), 
                 'test'   : str(os.path.join(input_dir,'test.prototxt'))}
    files = change_files(templates,num_classes,output_dir)
    return files

def gen_proto_files_by_num_classes(input_dir,output_dir,num_classes):
    templates = {'solver' : str(os.path.join(input_dir,'solver.prototxt')),
                 'train'  : str(os.path.join(input_dir,'train.prototxt')), 
                 'test'   : str(os.path.join(input_dir,'test.prototxt'))}
    files = change_files(templates,num_classes,output_dir)
    return files

