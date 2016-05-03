# --------------------------------------------------------
# Fast R-CNN obj_detector
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

import sys
from fast_rcnn.config import cfg, get_output_dir
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle as pickle
import heapq
from utils.blob import im_list_to_blob
import os
import scipy.io as sio
import scipy.sparse
from selective_search.calc_wrapper import calc_selective_search
#guyr_debug-svm from sklearn not needed
#from sklearn import svm

class detector(object):

    def __init__(self,net_path=None,weight_path=None,class_path=None,gpu_mode=True,gpu_id=0,use_svm=False,output_all_boxes=False,svm_path=None):
        self.net_path = net_path
        self.weight_path = weight_path
        self.svm_path = svm_path
        self.gpu_mode = gpu_mode
        self.gpu_id = gpu_id
        self.use_svm = use_svm
        self.output_all_boxes = output_all_boxes
        self.net = []
        self.sigmoid_data = []
        self.classes = self.get_class_names(class_path)
	self.num_classes = len(self.classes)
        self.configure_caffe()
        self.images = []
        self.roidb = []
        self.roidb_path = []
        self.detections = []

    def load_svm_data(self):
        #debug!!!! - remove later
        if not self.svm_path:
            print 'Warning - svm in new format does not exist.Returning'
            return

        svm_file_name = os.path.join(self.svm_path,'svm.pkl')
        print 'loading svm parameters : {:s}'.format(svm_file_name)
        [w,b] = pickle.load(open(svm_file_name, "rb"))
        self.net.params['cls_score'][0].data[...] = w
        self.net.params['cls_score'][1].data[...] = b
        
    def load_sigmoid(self):
        #debug!!!! - remove later
        if not self.svm_path:
            print 'Warning - svm in new format does not exist.Returning'
            return {'A' : np.zeros((1,self.num_classes)) ,'B' : np.zeros((1,self.num_classes))}

        sigmoid_file_name = os.path.join(self.svm_path,'sigmoid.pkl')
        print 'loading sigmoid parameters : {:s}'.format(sigmoid_file_name)
        data = pickle.load(open(sigmoid_file_name, "rb"))
        A = np.zeros((1,self.num_classes))
        B = np.zeros((1,self.num_classes))
        calib_s = np.zeros((1,self.num_classes))
        calib_b = np.zeros((1,self.num_classes))
        for c in range(1,self.num_classes):
            A[0,c]       = data[c][0]
            B[0,c]       = data[c][1]
            calib_s[0,c] = data[c][2]
            calib_b[0,c] = data[c][3]
        return {'A' : A ,'B' : B,'calib_s' : calib_s, 'calib_b' : calib_b}

    def set_roidb(self,roidb):
        self.roidb = roidb

    def get_class_names(self,class_path):
	try:
	    f = open(class_path,'r')
	except IOError:
    	    print('Failed to read labels file {}. Exiting'.format(class_path))
    	    sys.exit()
	content = f.readlines()
	f.close()
	
	labels = ['__background__']#label 0 is always background by default
	for s in content:
	    s_clean = s.strip().lower()
	    if s_clean!='' or s_clean!='\n':
	        labels.append(s_clean)
	return tuple(labels)

    def configure_caffe(self):
         assert os.path.exists(self.net_path)   , 'net file does not exist: {}'.format(self.net_path)
         assert os.path.exists(self.weight_path), 'net file does not exist: {}'.format(self.weight_path)
         if self.gpu_mode:
             caffe.set_mode_gpu()
             caffe.set_device(self.gpu_id)
         else:
             caffe.set_mode_cpu()
         net = caffe.Net(self.net_path, self.weight_path, caffe.TEST)
         net.name = os.path.splitext(os.path.basename(self.net_path))[0]
         self.net = net
         if self.use_svm:
             self.load_svm_data()
             self.sigmoid_data = self.load_sigmoid()
             cfg.DEDUP_BOXES = 0            
   
    @staticmethod
    def _get_image_blob(im):
        """Converts an image into a network input.

        Arguments:
            im (ndarray): a color image in BGR order

        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    @staticmethod
    def _project_im_rois(im_rois, scales):
        """Project image RoIs into the image pyramid built by _get_image_blob.

        Arguments:
            im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
            scales (list): scale factors as returned by _get_image_blob

        Returns:
            rois (ndarray): R x 4 matrix of projected RoI coordinates
            levels (list): image pyramid levels used by each projected RoI
        """
        im_rois = im_rois.astype(np.float, copy=False)

        if len(scales) > 1:
            widths = im_rois[:, 2] - im_rois[:, 0] + 1
            heights = im_rois[:, 3] - im_rois[:, 1] + 1

            areas = widths * heights
            scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
            diff_areas = np.abs(scaled_areas - 224 * 224)
            levels = diff_areas.argmin(axis=1)[:, np.newaxis]
        else:
            levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

        rois = im_rois * scales[levels]

        return rois, levels

    @staticmethod
    def _get_rois_blob(im_rois, im_scale_factors):
        """Converts RoIs into network inputs.

        Arguments:
            im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
            im_scale_factors (list): scale factors as returned by _get_image_blob

        Returns:
            blob (ndarray): R x 5 matrix of RoIs in the image pyramid
        """
        rois, levels = detector._project_im_rois(im_rois, im_scale_factors)
        rois_blob = np.hstack((levels, rois))
        return rois_blob.astype(np.float32, copy=False)
    
    @staticmethod
    def _get_blobs(im, rois):
        """Convert an image and RoIs within that image into network inputs."""
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = detector._get_image_blob(im)
        blobs['rois'] = detector._get_rois_blob(rois, im_scale_factors)
        return blobs, im_scale_factors

    @staticmethod
    def _bbox_pred(boxes, box_deltas):
        """Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
        heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = box_deltas[:, 0::4]
        dy = box_deltas[:, 1::4]
        dw = box_deltas[:, 2::4]
        dh = box_deltas[:, 3::4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    @staticmethod
    def _clip_boxes(boxes, im_shape):
        """Clip boxes to image boundaries."""
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
        return boxes

    def im_detect_int(self, im, boxes):
    	"""Detect object classes in an image given object proposals.

    	Arguments:
            im (ndarray): color image to test (in BGR order)
            boxes (ndarray): R x 4 array of object proposals

        Returns:
            scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
            boxes (ndarray): R x (4*K) array of predicted bounding boxes
        """
        blobs, unused_im_scale_factors = detector._get_blobs(im, boxes)

        # When mapping from image ROIs to feature map ROIs, there's some aliasing
        # (some distinct image ROIs get mapped to the same feature ROI).
        # Here, we identify duplicate feature ROIs, so we only compute features
        # on the unique subset.
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)

            blobs['rois'] = blobs['rois'][index, :]
            boxes = boxes[index, :]

        # reshape network inputs
        net = self.net
        net.blobs['data'].reshape(*(blobs['data'].shape))
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

        blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                                rois=blobs['rois'].astype(np.float32, copy=False))
        if self.use_svm:
            # use the raw scores before softmax under the assumption they
            # were trained as linear SVMs,convert to probabilities  
            svm_output = net.blobs['cls_score'].data
            #debug - add calibration
            svm_output = svm_output * self.sigmoid_data['calib_s'] + self.sigmoid_data['calib_b']
            scores = 1. / (1. + np.exp(svm_output * self.sigmoid_data['A'] + self.sigmoid_data['B'])) 
        else:
            # use softmax estimated probabilities
            scores = blobs_out['cls_prob']

        if cfg.TEST.BBOX_REG and not self.use_svm:
            # Apply bounding-box regression deltas
            box_deltas = blobs_out['bbox_pred']
            pred_boxes = detector._bbox_pred(boxes, box_deltas)
            pred_boxes = detector._clip_boxes(pred_boxes, im.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if cfg.DEDUP_BOXES > 0:
            # Map scores and predictions back to the original set of boxes
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

        return scores, pred_boxes

    @staticmethod
    def apply_nms(all_boxes, thresh,intra_class_nms=False):
        """Apply non-maximum suppression to all predicted boxes output."""
        num_classes = len(all_boxes)
        num_images = len(all_boxes[0])
        nms_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(num_classes)]
	for im_ind in xrange(num_images):
            for cls_ind in xrange(num_classes):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                if not 'keep_box_all_class' in vars():
                    dets_aug = dets
                else:
                    dets_aug = np.row_stack((keep_box_all_class,dets))
                keep = nms(dets_aug, thresh)
                if len(keep) == 0:continue
                if intra_class_nms:
                    keep_box_all_class = dets_aug[keep, :].copy()
                else:
                    nms_boxes[cls_ind][im_ind] = dets_aug[keep, :].copy()
            
            if intra_class_nms:
                #run over all classes to match image with class
                keep_set = set([tuple(x) for x in keep_box_all_class])
                for cls_ind in xrange(num_classes):
                    class_set = set([tuple(x) for x in all_boxes[cls_ind][im_ind]])
                    nms_boxes[cls_ind][im_ind] = np.array([x for x in class_set & keep_set]).copy()
                del keep_box_all_class
           
        return nms_boxes
   
    def get_images_list(self,images_path):
        if type(images_path) is list:
            return images_path
        elif type(images_path) is str:
            with open(images_path) as f:
                lines = f.read().splitlines()
                f.close()
            return lines
        else:
            raise

    def detect_single_aux(self, image_path,boxes):
        """aux function for detection of a single image,given the RoIs
        Arguments:
                images_path: color image to detect
                boxes: array of regions to detect

            Returns:
                scores (ndarray): R x K array of object class scores (K includes
                background as object category 0)
                boxes (ndarray): R x (4*K) array of predicted bounding boxes
        """      
        num_classes = self.num_classes
        all_boxes = [[] for _ in xrange(num_classes)]
        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}
        im = cv2.imread(image_path)
        assert(im is not None),'Failed to read image {}'.format(image_path)
        _t['im_detect'].tic()
        scores, boxes = self.im_detect_int(im, boxes)
        _t['im_detect'].toc()

        _t['misc'].tic()
        for j in xrange(1, num_classes):
            cls_scores = scores[:, j]
            cls_boxes = boxes[:, j*4:(j+1)*4]
            #NO confidence sorting is performed
            all_boxes[j] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
            assert all_boxes[j].shape[0]>0, '0 detections for class {}'.format(self.classes[j])

        _t['misc'].toc()
        print 'im_detect: {:.3f}s {:.3f}s'.format(_t['im_detect'].average_time,_t['misc'].average_time)
        self.detections = all_boxes


    def detect(self, images_path,roidb_path=None):
        """Main function for detection
        Arguments:
                images_path (list of image paths): color image to detect
                roidb_path [optional](path for region proposals for each image)

            Returns:
                scores (ndarray): R x K array of object class scores (K includes
                background as object category 0)
                boxes (ndarray): R x (4*K) array of predicted bounding boxes
        """
        self.images = self.get_images_list(images_path)
        self.roidb_path = roidb_path
        num_images = len(self.images)
        num_classes = self.num_classes
        # heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 100
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(num_classes)]

        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}

        if not self.roidb:
            self.roidb = self.load_selective_search_roidb()
        for i in xrange(num_images):
            im = cv2.imread(self.images[i])
            assert(im is not None),'Failed to read image {}'.format(self.images[i])
            _t['im_detect'].tic()
            boxes = self.roidb[i]['boxes']
            scores, boxes = self.im_detect_int(im, boxes)
            _t['im_detect'].toc()

            _t['misc'].tic()
            for j in xrange(1, num_classes):
                inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                #note that for all boxes output, NO confidence sorting is performed
                if not self.output_all_boxes:
                    top_inds = np.argsort(-cls_scores)[:max_per_image]
                    cls_scores = cls_scores[top_inds]
                    cls_boxes = cls_boxes[top_inds, :]
                all_boxes[j][i] = \
                        np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                assert all_boxes[j][i].shape[0]>0, '0 detections for class {} in image {}'.format(self.classes[j],i)

            _t['misc'].toc()

            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                  .format(i + 1, num_images, _t['im_detect'].average_time,_t['misc'].average_time)

        if not self.output_all_boxes:
            print 'Applying NMS to all detections'
            nms_dets = detector.apply_nms(all_boxes, cfg.TEST.NMS)
            self.detections = nms_dets
        else:
            self.detections = all_boxes

    def vis_detections(self, thresh=0.3,out_dir=None):
        """Visual debugging of detections."""
        import matplotlib.pyplot as plt
        if out_dir:
            if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        num_images = len(self.images)
        for i in xrange(num_images):
            im = cv2.imread(self.images[i])
            im = im[:, :, (2, 1, 0)]
            plt.cla()
            plt.imshow(im)

            for c in range(1,self.num_classes):
                det = self.detections[c][i]
	        if not len(det): continue
                for j in xrange(np.minimum(10, det.shape[0])):
                    bbox = det[j, :4]
                    score = det[j, -1]
                    if score > thresh:        
                        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                                           bbox[2] - bbox[0],
                                                           bbox[3] - bbox[1], fill=False,
                                                           edgecolor='g', linewidth=3))
                        x = (bbox[0] + bbox[2])/2
                        y = bbox[1]
                        s = '{}  {:.3f}'.format(self.classes[c], score)
                        plt.text(x, y, s, fontsize=14,horizontalalignment='center',weight='bold',backgroundcolor=(1,1,1))
            if out_dir:
                plt.savefig(os.path.join(out_dir,str(i)+'.jpg'))
            else:
                plt.show()

    def generate_detection_db(self,conf_th,top_k=None):
        db = []
        for im_ind in range(len(self.images)):
            #debug - follow db generation pace
            print 'saving detection db for image {:d}/{:d}'.format(im_ind+1,len(self.images))
            path = self.images[im_ind]
            boxes = []
            classes = []
            conf = []
            for cls_ind in range(self.num_classes):
                dets = self.detections[cls_ind][im_ind]
                cls_boxes = [];cls_conf = [];cls_classes = []
                if not len(dets): continue
                for d in dets:
                    if d[-1]<conf_th: continue
                    cls_boxes.append(d[:4])
                    cls_conf.append(d[-1])
                    cls_classes.append(cls_ind)
                if top_k:
                    order = np.argsort(-cls_conf)[:top_k]
                    cls_boxes = cls_boxes[order]
                    cls_conf = cls_conf[order]
                    cls_classes = cls_classes[order]
                #concat to detections
                boxes    = boxes   + cls_boxes
                conf     = conf    + cls_conf
                classes  = classes + cls_classes
                     
            data = {'image_path'  : path,
	            'boxes'       : boxes,
                    'confidence'  : conf,
		    'classes'     : classes}
            db.append(data)
        return tuple(db)

    def generate_detection_db_compact(self):
        '''
        Generates db assuming that all detections are valid, and no bbox regression
        '''
        db = []
        for im_ind in range(len(self.images)):
            path = self.images[im_ind]
            num_boxes = len(self.detections[1][im_ind])
            detections = np.zeros((num_boxes,self.num_classes + 4),np.float32)
            for box_idx in range(num_boxes):
                box = self.detections[1][im_ind][box_idx][:4]
                conf = [0.]#dummy detection of bg class
                for cls_ind in range(1,self.num_classes):
                    conf.append(self.detections[cls_ind][im_ind][box_idx][-1])
                detections[box_idx,:] = np.concatenate((np.array(conf,dtype=np.float32),box))

            data = {'image_path'  : path,
	            'detections'  : detections}
            db.append(data)
        return tuple(db)
                
    def load_selective_search_roidb(self):
        if self.roidb_path is not None: 
            assert os.path.exists(self.roidb_path), 'Selective search data not found at: {}'.format(filename)
        else:
            #write images to file
            images_file = os.getcwd()  + '/tempImages.txt'
            f = open(images_file,'w')
            for item in self.images:
               f.write("%s\n" % item)
            f.close()
            mat_file = os.getcwd()  + '/tempMat.mat'
            calc_selective_search(images_file,mat_file)
            os.remove(images_file)
            self.roidb_path = mat_file

        raw_data = sio.loadmat(self.roidb_path)['boxes'].ravel()
        num_images = len(self.images)
        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list)

    def create_roidb_from_box_list(self, box_list):
        num_images = len(self.images)

        assert len(box_list) == num_images, 'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32) 
            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({'boxes' : boxes,
                          'gt_classes' : np.zeros((num_boxes,),dtype=np.int32),
                          'gt_overlaps' : overlaps,
                          'flipped' : False})
        return roidb
   
    

