clear; clc

ROOT = getenv('NN_ROOT');
datadir = [ROOT '/data/rawdata/data_by_var/'];


s1 = load([datadir 'train/shot.mat']).shot;
s2 = load([datadir 'val/shot.mat']).shot;
s3 = load([datadir 'test/shot.mat']).shot;
shots = [s1; s2; s3];

t1 = load([datadir 'train/time.mat']).time;
t2 = load([datadir 'val/time.mat']).time;
t3 = load([datadir 'test/time.mat']).time;
times = [t1; t2; t3];

uniqshots = sort(unique(shots));

idx = [];
for i = 1:length(uniqshots)  
  j = find(shots==uniqshots(i));  
  [~, k] = sort(times(j));    
  idx = [idx; j(k)];      
  if ~issorted(times(j(k)))
    warning(num2str(i))
  end
end

%%

filelist = dir([datadir 'test']);
filelist = filelist(~startsWith({filelist.name}, 'd'));
filelist(1:2) = [];

i = find(strcmp( {filelist.name}, 'shape_params.mat'));
filelist(i) = [];

for i = 1:length(filelist)  
  i
  filename = filelist(i).name;
  varname = filename(1:end-4);
  
  x1 = load([datadir 'train/' filename]).(varname);
  x2 = load([datadir 'val/' filename]).(varname);
  x3 = load([datadir 'test/' filename]).(varname);
    
  x = [x1; x2; x3];
  x = x(idx,:,:);
  x = double(x);
  
  S = struct;
  S.(varname) = x;
  save([datadir '/all_eqnet/' filename], '-struct', 'S', '-v7.3');
    
end



%%
if 0
  clear; clc; close all

  ROOT = getenv('NN_ROOT');
  datadir = [ROOT '/data/rawdata/data_by_var/'];

  fds = {'coil_currents', 'IOH', 'IPF1AU', 'IPF2U', 'IPF3U', 'IPF5', ...
    'IPF3L', 'IPF2L', 'IPF1AL'};

  for i = 1:length(fds)
    load([datadir '/all_eqnet/' fds{i} '.mat']);
  end

  coil_currents_meas = [IOH IPF1AU IPF2U IPF3U IPF5 IPF3L IPF2L IPF1AL];

  if 0
    save('coil_currents_meas', 'coil_currents_meas', '-v7.3')
  end

end

% i = 8;
% j = 12000:14000;
% figure
% hold on
% plot(coil_currents_meas(j,i))
% plot(coil_currents(j,i), '--')





































