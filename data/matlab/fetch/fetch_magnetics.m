clear; clc; close all

load('magnetics_mdsnames.mat');
DATA_ROOT = getenv('DATA_ROOT');

% SETTINGS
mode = 'all_eqnet';
% mode = 'val';
% mode = 'test';
% signal_name = 'bpsignals'; 
% signal_name = 'ivsignals';
% signal_name = 'flsignals'; 
signal_name = 'lvsignals';
saveit = 0;
smoothit = 0;
plotit = 0;
datadir = [DATA_ROOT '/rawdata/data_by_var/' mode '/'];

% initialize
trees = magnetics.(signal_name).tree;  
tags = magnetics.(signal_name).tag;
shots = load([datadir 'shot.mat']).shot;
times = load([datadir 'time.mat']).time;
shotlist = unique(shots);
signal = zeros(length(shots), length(tags));
badshots = [];

% fetch data loop
for ishot = 1:length(shotlist)
  
  ishot / length(shotlist)
  
  shot = shotlist(ishot)  
  i = find(shots==shot);
  fetch_times = times(i);
  
  for j = 1:length(tags)
    try
      sig = mds_fetch_signal(shot, trees{j}, fetch_times, tags{j}, plotit, smoothit);            
      signal(i,j) = sig.sigs;
    catch
      signal(i,j) = nan;
      badshots(end+1) = shot;
      str = ['Could not fetch ' num2str(shot) ': ' tags{j}];
      warning(str)       
    end
  end
  
end

badshots = unique(badshots);

signal = double(signal);
if saveit
  S.(signal_name) = signal;
  save([datadir signal_name '.mat'], '-struct', 'S', '-v7.3');
end













