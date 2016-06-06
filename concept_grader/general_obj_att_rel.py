#! /usr/bin/env python

class general_obj_att_rel(object):
	'''
	This class implements general api to object,attribute and relationship detectors
	It extracts information from relevant modules
	This mainly holds as a place holder for more specific implementations
	'''
	def __init__(self):
		self._object = None
		self._attribute = None
		self._relationship = None
		raise NotImplemented

	@property
	def object_names(self):
		return self._object.class_names()
	
	@property
	def attribute_names(self):
		return self._attribute.class_names()

	@property
	def relationship_names(self):
		return self._relationship.class_names()

	def get_object_detector_data(self,images,boxes=None):
		'''
		Run object detector on a list of images
		In: images - a list of image paths to process
		    boxes - possibly object proposals to run on, if None the detector will generate the proposals
		Out: detections - a list of object detections, 1 per image
		'''
		detections,boxes = self._object.run_model(images,boxes)
		return (detections,boxes)

	def get_attribute_detector_data(self,images,boxes=None):
		'''
		Run attribute detector on a list of images
		In: images - a list of image paths to process
		    boxes - possibly object proposals to run on, if None the detector will generate the proposals
		Out: detections - a list of object detections, 1 per image
		'''
		detections = self._attribute.run_model(images,boxes)
		return detections

	def get_relationship_data(self,boxes,rel_idx=None,box_obj_idx=None):
		'''
		Run spatial model to between pairwise boxes
		In: boxes   - 2d array of bounding boxes, shape Nx4
                    rel_idx - indices of requested relationship to run(optional)
                    box_obj_idx - object assigned to each bounding box(optional)
		Out: a list per relationship, each element is a Nx(N-1)/2 array
             where each element is an array shape #relationshipsX2 (x>rel>y,x<rel<y)
		'''
                #TODO-should we add object indices to the boxes? this is coupling between the object detector
                #and the relationship... but the current implementation has partial dependency in 
                detections = self._relationship.run_model(boxes,rel_idx,box_obj_idx)
		return detections
	

