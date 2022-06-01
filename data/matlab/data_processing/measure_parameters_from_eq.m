clear all; clc; close all;
NN_ROOT = getenv('NN_ROOT');

saveit = 0;
plotit = 0;
mode = 'train';
shotnum = 204962;  % use [] for all shots

load('nstxu_obj_config2016_6565.mat')
struct_to_ws(tok_data_struct);
xlim = limdata(2,:);
ylim = limdata(1,:);

mpp = load('coneqt_nstxu_obj_config2016_6565.mat').tok_data_struct.mpp;
mppi = inv(mpp);

datadir = [NN_ROOT 'data/rawdata/data_by_var/' mode '/'];
vars = {'rbbbs', 'zbbbs', 'psirz', 'psibry', 'shot', 'time'};
for i = 1:length(vars)
  load([datadir vars{i} '.mat'])
end

if ~isempty(shotnum)
  i = shotnum==shot;
  rbbbs = rbbbs(i,:);
  zbbbs = zbbbs(i,:);
  psirz = psirz(i,:,:);
  psibry = psibry(i,:);
  time = time(i,:);
end
  
N = size(rbbbs,1);
%%
tic
for k = 1:N
  k
  psizr = squeeze(psirz(k,:,:))' * (-2*pi);
  shp = eq_analysis(psizr, psibry(k), rbbbs(k,:), zbbbs(k,:), tok_data_struct, mppi, plotit);

  if k == 1
    fns = fieldnames(shp);
  end

  for i = 1:length(fns)
    shape_params.(fns{i})(k) = shp.(fns{i});
  end
  
end
toc

if saveit
  save([datadir 'shape_params.mat'], 'shape_params');
end


%%
% A = shape_params;
% fds = fieldnames(A);
% for i = 1:length(fds)
%   A.(fds{i}) = A.(fds{i})(:);
% end
% 
% A = cell2struct(struct2cell(A), strcat('shape_', fieldnames(A)));
% struct_to_ws(A);
% fds = fieldnames(A);
% for i = 1:length(fds)
%   save([fds{i} '.mat'], fds{i}, '-v7.3');
% end
























