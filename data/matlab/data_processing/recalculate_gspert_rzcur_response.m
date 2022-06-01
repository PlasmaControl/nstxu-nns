clear all; clc; close all;
NN_ROOT = getenv('NN_ROOT');

mode = 'train';
saveit = 1;

datadir = [NN_ROOT 'data/rawdata/data_by_var/' mode '/'];

load('nstxu_obj_config2016_6565.mat')
struct_to_ws(tok_data_struct);
mpp = load('coneqt_nstxu_obj_config2016_6565.mat').tok_data_struct.mpp;
mppi = inv(mpp);
circ = nstxu2016_circ(tok_data_struct);
P = circ.Pxx(1:end-1,1:end-1);
rlim = limdata(2,:);
zlim = limdata(1,:);

shot = load([datadir 'shot']).shot;
ip = load([datadir 'ip']).ip;

gs_dzcurdix = load([datadir 'dzdis']).dzdis * P;
gs_drcurdix = load([datadir 'drdis']).drdis * P;
drcurdix = gs_drcurdix * 0;
dzcurdix = gs_drcurdix * 0;


for icoil = 1:53
  icoil

  s = ['dpsidix_coil' num2str(icoil)];
  dpsipladicoil = load([datadir s]).(s);
  
  dcphidicoil = dpsipladicoil * mppi;
  
  N = length(shot);
  for i = 1:N
    dcphi = reshape(dcphidicoil(i,:), 65, 65);
    dzcurdix(i,icoil) = sum(sum(dcphi .* zgg)) / ip(i);
    drcurdix(i,icoil) = sum(sum(dcphi .* rgg)) / ip(i);
  end
end


A = variables2struct(drcurdix, dzcurdix);

if saveit  
  A = cell2struct(struct2cell(A), strcat('shape_', fieldnames(A)));
  struct_to_ws(A);
  fds = fieldnames(A);
  for i = 1:length(fds)
    save([datadir fds{i} '.mat'], fds{i}, '-v7.3');
  end  
end


% plot comparison
shotlist = unique(shot);
isample = find(shot== shotlist(1));

figure
subplot(121)
hold on
plot(gs_drcurdix(isample,icoil))
plot(drcurdix(isample,icoil), '--')
subplot(122)
hold on
plot(gs_dzcurdix(isample,icoil))
plot(dzcurdix(isample,icoil), '--')


















































