#!/usr/bin/env python
# --------------------------------------------------------
# Selective Search Python wrapper for matlab
# --------------------------------------------------------
import os,imp
paths = imp.load_source('', os.path.join(os.path.dirname(__file__),'../../','init_paths.py'))
import subprocess
import os.path as osp

if paths.USE_MATLAB_SELECTIVE_SEARCH==True:
	# We assume your matlab binary is in your path and called `matlab'.
	# If either is not true, just add it to your path and alias it as matlab, or
	# you could change this file.
	MATLAB = 'matlab'

	# http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
	def _which(program):
		import os
		def is_exe(fpath):
			return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

		fpath, fname = os.path.split(program)
		if fpath:
		    if is_exe(program):
		        return program
		else:
		    for path in os.environ["PATH"].split(os.pathsep):
		        path = path.strip('"')
		        exe_file = os.path.join(path, program)
		        if is_exe(exe_file):
		            return exe_file

		return None

	if _which(MATLAB) is None:
		msg = ("MATLAB command '{}' not found. "
		       "Please add '{}' to your PATH.").format(MATLAB, MATLAB)
		raise EnvironmentError(msg)

	def calc_selective_search(images_file,mat_file):
		"""Running Matlab Function for Selective Search"""
		path = osp.join(paths.FAST_RCNN_BASE,'selective_search')
		data_path = osp.dirname(images_file)
		cmd = 'cd {} && '.format(path)
		cmd += '{:s} -nodisplay -nodesktop '.format(MATLAB)
		cmd += '-r "dbstop if error; '
		cmd += 'runMatlabSelectiveSearch(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"'.format(images_file,mat_file,data_path)
		print('Running:\n{}'.format(cmd))
		status = subprocess.call(cmd, shell=True)

else:
	def calc_selective_search(images_file,mat_file):
		return NotImplemented
		#TODO-continue and match to fast-rcnn format
		#img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

if __name__ == '__main__':
    #debug flow
    calc_selective_search()
