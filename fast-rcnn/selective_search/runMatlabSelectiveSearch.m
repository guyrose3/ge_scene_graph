function runMatlabSelectiveSearch(imageList,output_file,imagePath)
	addpath('SelectiveSearchCodeIJCV');
	addpath('SelectiveSearchCodeIJCV/Dependencies');

	fid = fopen(imageList,'r');
        assert(fid>=0,'Failed opening images file');

	[~,numFiles] = system(['python count.py ' imageList]);
	numFiles = str2num(numFiles);
	boxes = cell(1,numFiles);
        names = cell(1,numFiles);
        
	display(['Processing ' num2str(numFiles) ' files'])
	for idx=1:numFiles
   		tline = strtrim(fgets(fid));
   		[~,name,ext] = fileparts(tline);
		display(['Processing image: ' [name ext ' '] num2str(idx)]);
		try
                        p = [imagePath filesep tline];
                        if ~exist(p, 'file'), p = tline; end
   			I = imread(p);
                	%some default params for the selective search
   			boxes{idx} = selective_search_boxes(I,1,512);
                        names{idx} = tline;
		catch err
        		rethrow(err);
		end
	end
	fclose(fid);
	try
		save(output_file,'boxes','names');
	catch err
		throw(err);
	end

	display('Done');
end
