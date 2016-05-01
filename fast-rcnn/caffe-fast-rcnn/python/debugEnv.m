close all
clear all
clc

%debug
display('strart');

%enviroment defines
caffeRoot      = '~/fast-rcnn/caffe-fast-rcnn';
toolRoot       = '/home/guyrose3/fineTuningTool';
addpath([caffeRoot filesep 'matlab']);
addpath(caffeRoot);

%parameters
%config = struct() TODO - fill configuration - train+val parameters mostly
netName = 'caffeNet';



templateSolver = [toolRoot filesep 'templates/templateSolver.prototxt'];
templateNet    = [toolRoot filesep 'preTrainedNets' filesep netName filesep 'netTrainVal.prototxt'];
weights        = [toolRoot filesep 'preTrainedNets' filesep netName filesep 'weights.caffemodel'];


%inputs
%templateSolver
%templateNet 

%TODO - generate new trainVal file:
%first parse net.prototxt to struct, change fields, and generate new file

%cpu usage
caffe.set_mode_cpu();

%get net
net = struct('train','','val','');
net.train = caffe.get_net(templateNet,weights,'train');

%generate solver
%solverPath = templateSolver; %TODO - solver generator according to params 
%solver = caffe.get_solver(solverPath);



%debug
display('end');

