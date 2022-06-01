% ADJUST THIS TO REFLECT YOUR INSTALL LOCATION:
NN_ROOT = getenv('NN_ROOT'); 
DATA_ROOT = [NN_ROOT '/data/'];


restoredefaultpath

addpath(genpath([NN_ROOT '/eqnet/eval/']));
addpath(genpath([DATA_ROOT 'matlab/']));
addpath(genpath([DATA_ROOT 'rawdata/']));
setenv('DATA_ROOT', DATA_ROOT);

% Remove from path, any files in a directory labeled 'old'
% (Convenient for development to avoid path conflicts -- one can backup a
% copy with the same filename by saving it in any directory called old.) 
cd([DATA_ROOT 'matlab/'])
d = dir('**/old*');
for i = 1:length(d)
  rmpath([d(i).folder '/' d(i).name]);
end


% Add mds to path
MDSPLUS_DIR = '/usr/local/mdsplus';
if isfolder(MDSPLUS_DIR)
  setenv('MDSPLUS_DIR', MDSPLUS_DIR);
  addpath(genpath(MDSPLUS_DIR));
else
  warning('MDSPLUS install not found');
end
  

