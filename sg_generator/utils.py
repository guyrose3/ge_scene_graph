import os,imp
paths = imp.load_source('..', os.path.join(os.path.dirname(__file__),'init_paths.py'))
import numpy as np

###########################
#helper functions
###########################
def bbox_IoU(bb1,bb2):
    #calc bb overlap
    bi=[max(bb1[0],bb2[0]) , max(bb1[1],bb2[1]) , min(bb1[2],bb2[2]) , min(bb1[3],bb2[3])]
    iw=bi[2]-bi[0]+1
    ih=bi[3]-bi[1]+1
    if iw>0 and ih>0:
        #compute overlap as area of intersection / area of union
        ua=(bb1[2]-bb1[0]+1)*(bb1[3]-bb1[1]+1)+ \
           (bb2 [2]-bb2 [0]+1)*(bb2 [3]-bb2 [1]+1)- \
            iw*ih
        ov=iw*ih/ua
        return ov
    return 0.
