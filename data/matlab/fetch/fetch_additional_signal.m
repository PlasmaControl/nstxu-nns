clear all; clc; close all
mdsclose; mdsdisconnect;

% SETTINGS
mode = 'all_eqnet';
DATA_ROOT = getenv('DATA_ROOT');
datadir = [DATA_ROOT '/rawdata/data_by_var/' mode '/'];


% tags = {'IOH', 'IPF1AU', 'IPF2U', 'IPF3U', 'IPF5', 'IPF3L', 'IPF2L', 'IPF1AL'};
% tree = 'engineering';
% for j = 1:length(tags)
%   signal_name = tags{j};
%   mdstag = ['.ANALYSIS:' tags{j}];

% tree = 'efit01';
% mdstag = '.RESULTS.AEQDSK:ZCUR';
% signal_name = 'vloop';

tree = 'efit01';
mdstag = '.RESULTS.AEQDSK:CHISQ';
signal_name = 'chisq';

% tree = 'efit01';
% mdstag = '.RESULTS.AEQDSK:CMPR2';
% % mdstag = '.RESULTS.DERIVED:PABZBTZ0';

signal_name = 'cmpr2';


saveit = 0;
plotit = 0;

% FETCH THE DATA
shots = load([datadir 'shot.mat']).shot;
times = load([datadir 'time.mat']).time;
shotlist = unique(shots);
signal = [];

for i = 1:length(shotlist)
  
  i
  shot = shotlist(i)
  fetch_times = times(shots==shot);
  
  sig = mds_fetch_signal(shot, tree, fetch_times, mdstag, plotit);
  
  signal = [signal sig.sigs];
  
end

S = struct;
signal = double(signal(:));
if saveit
  if length(signal) == length(times) && ~any(isnan(signal))
    S.(signal_name) = signal;
    save([datadir signal_name '.mat'], '-struct', 'S', '-v7.3');
  else
    warning('Data did not fetch correctly.')
  end
end

% end











