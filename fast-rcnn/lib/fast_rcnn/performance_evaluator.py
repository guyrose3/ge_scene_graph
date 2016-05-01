#!/usr/bin/env python
# --------------------------------------------------------
# Class for evaluating localization results
# Written by Guy Rosenthal
# --------------------------------------------------------

from __future__ import division
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../../','init_paths.py'))
import datasets.general_dataset
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from fast_rcnn.config import cfg
import csv
import time

class performance_evaluator(datasets.general_dataset):
    def __init__(self,name, image_set, data_path,gt_file=None, detections=None,cfg=None):
        datasets.general_dataset.__init__(self,name, image_set, data_path)
        self.output_path = os.path.join(data_path,'evaluation',name + '.' + image_set + '.' + time.strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    	self.gt_file = os.path.join(self.output_path,'gt.pkl')
        self.image_list = data_path + '/' + image_set + '.txt'
	self.cfg = cfg
        self.gt = self.get_gt()
	self.check_legal_args()
        self.detector = []
	self.detections = detections
	#save the database if not existed
	if not os.path.exists(self.gt_file):
	    print 'saving ground truth database into {}'.format(self.gt_file)
	    self.write_gt_database()

    def get_detections(self):
        self.detector.set_roidb(self.roidb_handler())
        self.detector.detect(self.image_list)
        #consider all positives detections for ROC curve generation
        self.detections = self.detector.generate_detection_db(0.0) 

    def set_detector(self,detector):
        self.detector = detector    
	
    def check_legal_args(self):
	if self.image_list is not None:
            assert os.path.exists(self.image_list), 'Images list does not exist: {}'.format(self.image_list)
        #check if image list needs modificatios
        images = [line.rstrip('\n') for line in open(self.image_list)]
        if not os.path.exists(images[0]):
            data_path = os.path.dirname(self.image_list)
            images = [os.path.join(data_path,i) for i in images]
            self.image_list = os.path.join(self.output_path,'image_list.txt')
            f = open(self.image_list,'w') 
            for item in images:
                f.write("%s\n" % item)
            f.close()


    def write_gt_database(self):
	with open(self.gt_file, 'wb') as fp:
  	    cPickle.dump(self.gt, fp)

    def get_gt(self):
	if os.path.exists(self.gt_file):
	    with open(self.gt_file, 'rb') as fp:
		gt = cPickle.load(fp)
	else: 
	    gt = self.generate_gt_database()  
	return gt

    def generate_gt_from_file(self,xml_path):
	raise NotImplementedError

    def generate_gt_database(self):
	with open(self.image_list) as f:
	    imagesPath = f.readlines()
	gt = []
	for l in imagesPath:
	    index,ext = os.path.splitext(l.rstrip())#assuming that for each image the xml is in the same path
	    single_image_data = self._load_pascal_annotation(index)
	    converted_data = {'image_path'  : index + ext,
			      'boxes'       : single_image_data['boxes'],
			      'gt_classes'  : single_image_data['gt_classes'],
			      'gt_overlaps' : single_image_data['gt_overlaps']}
	    gt.append(converted_data)
	return tuple(gt)

    @staticmethod
    def auc(x, y, reorder=False):

        if x.shape[0] < 2:
            raise ValueError('At least 2 points are needed to compute'
                             ' area under curve, but x.shape = %s' % x.shape)

        direction = 1
        if reorder:
            # reorder the data points according to the x axis and using y to
            # break ties
            order = np.lexsort((y, x))
            x, y = x[order], y[order]
        else:
            dx = np.diff(x)
            if np.any(dx < 0):
                if np.all(dx <= 0):
                    direction = -1
                else:
                    raise ValueError("Reordering is not turned on, and "
                                     "the x array is not increasing: %s" % x)

        area = direction * np.trapz(y, x)

        return area
    
    def evaluate(self,confusion_th=0.3):
        tp = np.array([])
        fp = np.array([])
        classes = np.array([])
        gt_classes = np.array([])
        confidence = np.array([])
        npos = [0] * self.num_classes#number of gt bboxes
	#assuming detections format containing ground truth
	for d in self.detections:
    	    image_path = os.path.basename(d['image_path'])
	    gt = filter(lambda single_gt: os.path.basename(single_gt['image_path']) == image_path, self.gt)
            #should be only one registry with detections for each image
            assert len(gt)==1, 'More than 1 registries ({}) for image: {}'.format(str(len(gt)),image_path)
            gt = gt[0]     
	    metric = self.calc_metric(d,gt)
            tp = np.concatenate([tp,metric['tp']])
            fp = np.concatenate([fp,metric['fp']])
            classes = np.concatenate([classes,metric['classes']]) 
            gt_classes = np.concatenate([gt_classes,metric['gt_classes']])
            confidence = np.concatenate([confidence,d['confidence']])
            for c in gt['gt_classes']:
                npos[c]+=1
                        
	#sort by confidence
        order = np.argsort(-confidence)
        tp = tp[order]
        fp = fp[order]
        classes = classes[order]
        gt_classes = gt_classes[order]
        confidence = confidence[order]
        #compute tp percent vs. confidence level
        conf_grid = np.linspace(confidence.min(), confidence.max(), num=50)
        conf_idx = np.zeros(len(confidence),dtype=np.int32)
        for i in range(len(conf_idx)): conf_idx[i] = np.abs(conf_grid - confidence[i]).argmin()
        tp_perc = np.zeros(len(conf_grid)) 
        for i in range(len(tp_perc)):
            tp_in_range = tp[np.argwhere(conf_idx==i).ravel()]
            if tp_in_range.shape[0]>0:
                tp_perc[i] = tp_in_range.sum()/tp_in_range.shape[0]
        #compute Average Precision per class
        AP = [0] * self.num_classes#excluding background
        for c in range(1,self.num_classes):
            tpc = tp[np.where(classes == c)]
            fpc = fp[np.where(classes == c)]
            #handle case where not enough gt instances of a class are present
            if len(tpc)<2:
                AP[c] = 0
                print 'Not enough samples for {} class, setting precision to 0'.format(self.classes[c])
                continue
            tpc=np.cumsum(tpc)
            fpc=np.cumsum(fpc)
            rec=tpc / npos[c]
            prec=tpc / (fpc+tpc)
            AP[c] = performance_evaluator.auc(rec,prec)
        #compute Map
        tp=np.cumsum(tp)
        fp=np.cumsum(fp)
        rec=tp / np.sum(npos)
        prec=tp / (fp+tp)
        MaP = performance_evaluator.auc(rec,prec)
        #generate output
        self.write_conf_mat(classes[np.argwhere(confidence>confusion_th)].ravel(),gt_classes[np.argwhere(confidence>confusion_th)].ravel())
        self.output_evaluation(AP,MaP,npos,conf_grid,tp_perc)

    def write_conf_mat(self,classes,gt_classes):
        conf_mat = np.zeros((self.num_classes,self.num_classes))
        for i in range(len(classes)):
            row = classes[i]
            col = gt_classes[i]
            conf_mat[row,col]+=1
        #write to cvs file
        fl = open(os.path.join(self.output_path,'confusion.csv'), 'w')
        top = [' ']
        for l in self.classes:top.append(l)
        writer = csv.writer(fl)
        writer.writerow(top)
        for i in range(len(self.classes)):
            row = [self.classes[i]]
            for n in conf_mat[i,:]: row.append(n)
            writer.writerow(row)
        fl.close()       

    def output_evaluation(self,AP,MaP,npos,conf_grid,tp_perc):
        #confidence vs tp percent plot
        plt.plot(conf_grid,tp_perc)
        plt.xlabel('confidence')
        plt.ylabel('True positive %')
        plt.savefig(os.path.join(self.output_path,'conf_tp.png'))
        #Map graph
        ind = np.arange(self.num_classes-1)
        width = 0.5
        fig, ax = plt.subplots()
        rects = ax.bar(ind, AP[1:], width, color='b')
        ax.set_ylabel('Average Precision')
        ax.set_title('Mean Average Precision - {0:.2f}'.format(MaP))
        ax.set_xticks(ind)
        #tags = [str(i) for i in range(self.num_classes-1)]
        #ax.set_xticklabels(tags,horizontalalignment = 'center')
        plt.savefig(os.path.join(self.output_path,'AP.png'))
        #generate log
        f = open(os.path.join(self.output_path,'performance.txt'),'w')
        f.write('Mean Average Performance:{0:.2f}\n'.format(MaP))
        f.write('Class Performance:\n')
        for c in range(1,self.num_classes):
            f.write('{}: test bounding boxes:{}, Average precision:{} \n'.format(self.classes[c],npos[c],AP[c]))
        f.close()

    
    def calc_metric(self,d,gt):
        num_valid_detections = len(d['boxes'])
        tp = [0] * num_valid_detections
        fp = [0] * num_valid_detections
        classes = [0] * num_valid_detections
        gt_classes = [0] * num_valid_detections
        detected = [False] * len(gt['gt_classes'])

        for i in range(num_valid_detections):
	    bb_det = d['boxes'][i]
	    class_det = d['classes'][i]
            classes[i] = class_det
	    #check only against gt of the predicted class-only true and false positives are checked
            ovmax = 0
            #idx = np.argwhere(gt['gt_classes'] == class_det)
            idx = np.argwhere(gt['gt_classes'])
            clmax = 0#background class by default
          
	    for j in idx:
	        bb_gt = gt['boxes'][j][0].astype(float)
	        class_gt = gt['gt_classes'][j][0]#should be single class
		#calc bb overlap
                bi=[max(bb_det[0],bb_gt[0]) , max(bb_det[1],bb_gt[1]) , min(bb_det[2],bb_gt[2]) , min(bb_det[3],bb_gt[3])]
                iw=bi[2]-bi[0]+1
                ih=bi[3]-bi[1]+1
                if iw>0 and ih>0:
                    #compute overlap as area of intersection / area of union
                    ua=(bb_det[2]-bb_det[0]+1)*(bb_det[3]-bb_det[1]+1)+ \
                       (bb_gt [2]-bb_gt [0]+1)*(bb_gt [3]-bb_gt [1]+1)- \
                       iw*ih
                    ov=iw*ih/ua
                    if ov>ovmax:
                        ovmax=ov
                        jmax=j
                        clmax = class_gt
            #assign detection as true positive/don't care/false positive
            if ovmax>=cfg.TEST.GT_OVERLAP:
                if not detected[jmax] and clmax==class_det:
                    tp[i]=1 #true positive
		    detected[jmax]=True
                elif not clmax==class_det:fp[i]=1 #detected false class
                else: fp[i]=1 #false positive (multiple detection)
            else:
                fp[i]=1 #false positive
            gt_classes[i] = clmax

	return {'tp' : tp , 'fp' : fp, 'classes' : classes,'gt_classes' : gt_classes,'confidence' : d['confidence'][:num_valid_detections]}

    def vis_gt_boxes(self):
        """Visual debugging of detections."""
        import cv2
        num_images = len(self.gt)
        for i in range(num_images):
            im = cv2.imread(self.image_path_at(i))
            im = im[:, :, (2, 1, 0)]
            plt.cla()
            plt.imshow(im)
            gt_image = self.gt[i]
            for j in range(len(gt_image['boxes'])):
               bbox = gt_image['boxes'][j]
               c    = gt_image['gt_classes'][j]       
               plt.gca().add_patch(plt.Rectangle((float(bbox[0]), float(bbox[1])),
                                                  float(bbox[2]) - float(bbox[0]),
                                                  float(bbox[3]) - float(bbox[1]), fill=False,
                                                  edgecolor='r', linewidth=3))
               x = (bbox[0] + bbox[2])/2
               y = bbox[1]
               s = '{}'.format(self.classes[c])
               plt.text(x, y, s, fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
            plt.show()

