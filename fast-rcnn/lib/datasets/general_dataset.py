# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.general_dataset
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import shutil
import PIL
from selective_search.calc_wrapper import calc_selective_search
import time

class general_dataset(datasets.imdb):
    def __init__(self,name, image_set, data_path):
        datasets.imdb.__init__(self, name + '_' + image_set)
        self._image_set = image_set
        self._data_path = data_path#path where train.txt,test.txt,labels.txt are found
        self._classes = self.get_class_names()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'#TODO - expand to other image formats
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        # PASCAL specific config options TODO-check if can be removed
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}
        self.k_fold = None

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def gen_k_fold(self,k_fold):
        self.k_fold = k_fold
        if self.k_fold>1: 
            self.folds = self.get_k_fold_indices(k_fold)

    @property#overwrite base class
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(self._data_path, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def get_class_names(self):
	labels_file = self._data_path + '/labels.txt'
	try:
	    f = open(labels_file,'r')
	except IOError:
    	    print('Failed to read labels file {}. Exiting'.format(labels_file))
    	    raise
	content = f.readlines()
	f.close()
	
	labels = ['__background__']#label 0 is always background by default
	for s in content:
	    s_clean = s.strip().lower()
	    if s_clean!='' or s_clean!='\n':
	        labels.append(s_clean)
	return tuple(labels)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path,self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if not 'test' in self._image_set:
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            print 'Joining ground truth to roi set'
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..','selective_search_data', self.name + '.mat'))
        self.calculate_selective_search_data(filename)
        assert os.path.exists(filename), 'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def get_pending_images_file(self,mat_file,images_file):
        file_array = sio.loadmat(mat_file)['names'].ravel()
        done_files = []
        for i in range(file_array.shape[0]):
            a = file_array[i][0]
            done_files.append(str(a))
        all_files = [line.rstrip('\n') for line in open(images_file)]
        undone_files = list(set(all_files) - set(done_files))

        if 0==len(undone_files): return None
        #write image to file
        tmp_file = os.path.join(self._data_path,'tmp_images_file.txt')
        f=open(tmp_file,'w')
        for item in undone_files:
            f.write("%s\n" % item)
        f.close()
        return tmp_file

    def merge_mat_files(self,mat_file,tmp_mat_file):
        names_a = sio.loadmat(mat_file    )['names'].ravel()
        names_b = sio.loadmat(tmp_mat_file)['names'].ravel()
        names = np.concatenate((names_a,names_b),axis=0) 
        boxes_a = sio.loadmat(mat_file    )['boxes'].ravel()
        boxes_b = sio.loadmat(tmp_mat_file)['boxes'].ravel()
        boxes = np.concatenate((boxes_a,boxes_b),axis=0)
        sio.savemat(mat_file,{'names':names,'boxes':boxes})

    def write_final_mat_file(self,database_mat_file,mat_file,images_file):
        names = sio.loadmat(database_mat_file)['names'].ravel()
        boxes = sio.loadmat(database_mat_file)['boxes'].ravel()
        with open(images_file) as f:
            lines = f.read().splitlines()
            name_txt = [os.path.basename(l) for l in lines]
        indecies = []
        for n in name_txt:
            i = [index for index,name in enumerate(names) if os.path.basename(name[0]) == n]
            assert len(i)==1, 'Image appears more than once'
            indecies.append(i[0])
        sio.savemat(mat_file,{'names':names[indecies],'boxes':boxes[indecies]})
        
    def calculate_selective_search_data(self,mat_file):
        #generate unified mat file
        database_mat_file = os.path.join(self.cache_path, '..','selective_search_data', str.replace(self.name, '_' + self._image_set, '.mat'))
        if not os.path.exists(os.path.dirname(mat_file)):
            os.makedirs(os.path.dirname(mat_file))
        images_file = os.path.join(self._data_path,self._image_set + '.txt')
        #if the file don't exist,run all list
        if not os.path.exists(database_mat_file):
            calc_selective_search(images_file,database_mat_file)
            shutil.copyfile(database_mat_file, mat_file)
        #else needs to check, which images were already processed,remove them and run the rest
        else:
            tmp_images_file = self.get_pending_images_file(database_mat_file,images_file)
            if not (None==tmp_images_file):
                tmp_mat_file = os.path.join(self._data_path,'tmp_mat.mat')
                calc_selective_search(tmp_images_file,tmp_mat_file)
                self.merge_mat_files(database_mat_file,tmp_mat_file)
                os.remove(tmp_images_file)
                os.remove(tmp_mat_file)
            #write subset from database to the mat file to be used
            self.write_final_mat_file(database_mat_file,mat_file,images_file)
            
    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = self._data_path + '/' + os.path.splitext(index)[0] + '.xml'

        def get_data_from_tag(node, tag):              
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            cls = self._class_to_ind[str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        path = os.path.join('results', self._name)
	if not os.path.exists(path):
    	    os.makedirs(path)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + '/det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        #return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(detector.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        #comp_id = self._write_voc_results_file(all_boxes)
	self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def debug_output_gt_roidb(self,im,boxes,classes,polarity='Pos'):
        import cv2
        """
        save ground truth ROIs to images, for debug purposes.
        """
        root_dir = os.path.join(self._data_path,'debug_bb',polarity)
        for i in range(len(classes)):
            class_name = self.classes[classes[i]]
            class_dir = os.path.join(root_dir,class_name)
            if not os.path.exists(class_dir): os.makedirs(class_dir)
            filename = time.strftime("%Y%m%d-%H%M%S") + str(i) + '.jpg'   
            roi = im[boxes[i][1]:boxes[i][3],boxes[i][0]:boxes[i][2]]
            cv2.imwrite(os.path.join(class_dir,filename), roi)

    def get_k_fold_indices(self,k):
        gt_roidb = self.gt_roidb()
        assert len(gt_roidb)==self.num_images,'Number of gt annotations is different than number of images'
        k_indices = np.zeros((self.num_classes,self.num_images),dtype=np.int32)
        for cls_ind in xrange(1,self.num_classes):
            im_positives = np.zeros((self.num_images),dtype=np.int8)
            #collect positives
            for im_ind in xrange(self.num_images):
                im_positives[im_ind] = len(np.argwhere(gt_roidb[im_ind]['gt_classes']==cls_ind))
            #allocate to folds
            pos_ind = 0
            neg_ind = 0
            for im_ind in xrange(self.num_images):
                if im_positives[im_ind]>0:
                    k_indices[cls_ind,im_ind] = pos_ind % k
                    pos_ind+=1
                else:
                    k_indices[cls_ind,im_ind] = neg_ind % k
                    neg_ind+=1
            assert pos_ind>=k, 'not enough positives in class {:d}'.format(cls_ind)
               
        return k_indices  

    def append_flipped_images(self):#overwrite base class
        num_images = self.num_images
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]

        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1

            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2 
        #add k-fold support
        if self.k_fold: 
            self.folds = np.hstack((self.folds,self.folds))


if __name__ == '__main__':
    d = datasets.disney_dataset('train', 'addPathHere')
    res = d.roidb
    from IPython import embed; embed()
