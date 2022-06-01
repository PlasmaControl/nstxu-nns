% Reorganize data to store by variable name instead of by shot number. 
% (Allows easier handling of the datasets, since each variable can
% then fit into memory). 

clear; clc; close all;

mode = 'val';
DATA_ROOT = getenv('DATA_ROOT');

build_dir = [DATA_ROOT '/rawdata/data_by_shot/' mode '/'];
save_dir = [DATA_ROOT '/rawdata/data_by_var/' mode '/'];

if ~exist(save_dir, 'dir'), mkdir(save_dir); end

d = dir(build_dir);
d(1:2) = [];

for ishot = 1:length(d)
  tic
  disp(ishot)
  
  targs_fn = [d(ishot).folder '/' d(ishot).name];
  targs = load(targs_fn).targs; 

  if ishot == 1, fields = fieldnames(targs); end

  % reorganize data a bit
  nsamples = length(targs.ip);
  
  % dum = false(nsamples,1);
  % dum(targs.iuse_handlabeled) = true;
  % targs.iuse_handlabeled = dum;

  targs.shot(1:nsamples) = targs.shot;
  targs.shot_is_good(1:nsamples) = targs.shot_is_good;


  for ifield = 1:length(fields)
    field = fields{ifield};

    if strcmp(field, 'dpsidix_smooth15') 
      continue

    elseif contains(field, 'dpsidix')   

      for icoil = 1:54   
          fieldname = [field '_coil' num2str(icoil)];
          filename = [save_dir fieldname '.mat'];
          data = targs.(field)(:,:,icoil);
          append2mat(filename, data, fieldname, ishot);    
      end

    else

      filename = [save_dir field '.mat'];
      data = targs.(field);
      append2mat(filename, data, field, ishot);    

    end  
  end
  toc
end

function append2mat(filename, data, field, ishot)
  
  if ~isa(data, 'double'), data = double(data); end
  if size(data,1) == 1, data = data'; end 
  
  [dim1, dim2, dim3]  = size(data);
  
  m = matfile(filename, 'Writable', true);
  
  if ishot == 1, m.(field) = zeros(0,0,0); end
  
  N = size(m, field, 1);
  if dim3 == 1
    m.(field)(N+1:N+dim1, 1:dim2) = data;
  else
    m.(field)(N+1:N+dim1, 1:dim2, 1:dim3) = data;
  end
  
end




















































