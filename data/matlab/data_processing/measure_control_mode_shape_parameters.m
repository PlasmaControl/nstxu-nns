
clear all; clc; close all;

load('nstxu_obj_config2016_6565.mat')
mpp = load('coneqt_nstxu_obj_config2016_6565.mat').tok_data_struct.mpp;
mppi = inv(mpp);

struct_to_ws(tok_data_struct);
circ = nstxu2016_circ(tok_data_struct);
mpcx = mpc*circ.Pcc;

NN_ROOT = getenv('NN_ROOT');

load([NN_ROOT 'data/rawdata/data_by_var/test/dpsidix_coil13.mat']);
load([NN_ROOT 'data/rawdata/data_by_var/test/shape_rx_lo.mat']);
load([NN_ROOT 'data/rawdata/data_by_var/test/shape_zx_lo.mat']);
load([NN_ROOT 'data/rawdata/data_by_var/test/psirz.mat']);
load([NN_ROOT 'data/rawdata/data_by_var/test/drxdis.mat']);
load([NN_ROOT 'data/rawdata/data_by_var/test/dzxdis.mat']);



%%
load('/p/nstxusr/nstx-users/jwai/nstxu-nns/data/rawdata/data_by_var/test/shot.mat')

clc
clear dz dzdis

isample = find(shot==205062);

icoil = 13;

ip = load('/p/nstxusr/nstx-users/jwai/nstxu-nns/data/rawdata/data_by_var/val/ip.mat').ip(isample);

dcphi = dpsidix_coil13(isample,:) * mppi;

for i = 1:length(isample)  
  dz(i) = sum( dcphi(i,:)' .* zgg(:)) / ip(i);
end


dzdis = load([NN_ROOT 'data/rawdata/data_by_var/test/dzdis.mat']).dzdis;
dzdis = dzdis * circ.Pxx(1:end-1, 1:end-1);
dzdis = dzdis(isample,icoil)


figure
hold on
plot(dz)
plot(dzdis)

%%




















%%
% clc
% isample = [110];
% 
% psizr = squeeze(psirz(isample,:,:))';
% rx_lo = shape_rx_lo(isample);
% zx_lo = shape_zx_lo(isample);
% 
% figure
% hold on
% contour(rg,zg,psizr,20)
% scatter(rx_lo, zx_lo, 50, 'filled')
% axis equal
% 
% icoil = 2;
% 
% drxlodis = squeeze(drxdis(:,1,:)) * circ.Pxx(1:end-1,1:end-1);
% drxupdis = squeeze(drxdis(:,2,:)) * circ.Pxx(1:end-1,1:end-1);
% drxlodis = drxlodis(isample,icoil);
% drxupdis = drxupdis(isample,icoil);
% 
% dzxlodis = squeeze(dzxdis(:,1,:)) * circ.Pxx(1:end-1,1:end-1);
% dzxupdis = squeeze(dzxdis(:,2,:)) * circ.Pxx(1:end-1,1:end-1);
% dzxlodis = dzxlodis(isample,icoil);
% dzxupdis = dzxupdis(isample,icoil);
% 
% dpsi = reshape(dpsidix_coil2(isample,:), nz, nr)';
% 
% [~,drxlodis_gs,dzxlodis_gs] = bicubicHermite(rg,zg,dpsi,rx_lo,zx_lo);
% 
% dpsi_vac = reshape(mpcx(:,icoil), nz, nr);
% [~,drxlodis_vac,dzxlodis_vac] = bicubicHermite(rg,zg, dpsi_vac, rx_lo, zx_lo);
% 
% rx = [drxlodis
% drxlodis_gs
% drxlodis_vac]
% 
% zx = [dzxlodis
% dzxlodis_gs
% dzxlodis_vac]




























