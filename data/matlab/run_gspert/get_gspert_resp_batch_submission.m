% Submits the batch jobs to run gspert

clear all; clc; close all

mode = 'train';

DATA_ROOT = getenv('DATA_ROOT');

save_dir = [DATA_ROOT '/rawdata/data_by_shot/' mode '/'];
job_topdir = [DATA_ROOT '/matlab/run_gspert/jobs/' mode '/'];

if ~exist(job_topdir, 'dir'), mkdir(job_topdir); end   
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

  
split = load('train_test_val_split.mat').split;
shots = split.([mode 'shots']);


for ishot = 1:length(shots)
  
  shot = shots(ishot);
  
  % arguments for batch job
  job_args.save_fn = [save_dir 'response_' num2str(shot) '.mat'];
  job_args.shot = shot;
  job_args.DATA_ROOT = DATA_ROOT;
  job_args.MDSPLUS_DIR = getenv('MDSPLUS_DIR');
  job_args.saveit = 1; 
  
  % create job dir
  jobdir = [job_topdir num2str(shot) '/'];  
  if exist(jobdir,'dir'), rmdir(jobdir,'s'); end
  mkdir(jobdir)
  
  % copy args and script to jobdir
  save([jobdir 'job_args.mat'], 'job_args');
  jobscript = [DATA_ROOT '/matlab/run_gspert/get_gspert_resp_batch.m'];  
  copyfile(jobscript, jobdir);
  
  % submit job
  cd(jobdir)
  batchscript_fn =  [DATA_ROOT '/matlab/run_gspert/job.sbatch'];
  system(['sbatch ' batchscript_fn ' get_gspert_resp_batch.m']);    

end

cd(job_topdir)

