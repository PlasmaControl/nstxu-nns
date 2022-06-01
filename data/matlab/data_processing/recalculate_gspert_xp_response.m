clear all; clc; close all;
NN_ROOT = getenv('NN_ROOT');


mode = 'test';
saveit = 0;

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
drxdis = load([datadir 'drxdis']).drxdis;
dzxdis = load([datadir 'dzxdis']).dzxdis;
rx_lo = load([datadir 'shape_rx_lo']).shape_rx_lo;
rx_up = load([datadir 'shape_rx_up']).shape_rx_up;
zx_lo = load([datadir 'shape_zx_lo']).shape_zx_lo;
zx_up = load([datadir 'shape_zx_up']).shape_zx_up;
psirz = load([datadir 'psirz']).psirz;
psibry = load([datadir 'psibry']).psibry;

psizr = permute(psirz, [1 3 2]) * (-2*pi);

gs_drxlodix = squeeze(drxdis(:,1,:)) * P;
gs_drxupdix = squeeze(drxdis(:,2,:)) * P;
gs_dzxlodix = squeeze(dzxdis(:,1,:)) * P;
gs_dzxupdix = squeeze(dzxdis(:,2,:)) * P;

drxlodix = 0*gs_drxlodix;
drxupdix = 0*gs_drxlodix;
dzxlodix = 0*gs_drxlodix;
dzxupdix = 0*gs_drxlodix;

for icoil = 1:53
  icoil

  s = ['dpsidix_coil' num2str(icoil)];
  dpsipladicoil = load([datadir s]).(s);

  dpsiapp = [mpc mpv] * P;
  dpsiapp = reshape(dpsiapp(:,icoil), nz, nr); 

  N = length(shot);

  for i = 1:N

    dpsipla = reshape(dpsipladicoil(i,:), nz, nr);
    dpsi = dpsiapp + dpsipla;

    % ========
    % lower xp
    % ========
    r = rx_lo(i);
    z = zx_lo(i);

    rx_lo_ref = 0.55;
    zx_lo_ref = -1.1;
    zx_up_ref = 1.1;
    rx_up_ref = 0.55;

    validxp = ~isnan(r) &&  (norm([r-rx_lo_ref, z-zx_lo_ref]) < 0.5);

    if validxp
      psi = reshape(psizr(i,:,:), nz, nr);

      [~, dpsidr, dpsidz] = bicubicHermite(rg, zg, dpsi, r, z);
      [~, ~, ~, psi_rr, psi_zz, psi_rz] = bicubicHermite(rg, zg, psi, r, z);
      J = [psi_rr psi_rz; psi_rz psi_zz];
      dx = -inv(J)*[dpsidr; dpsidz];
    else
      dx = [nan nan];
      rx_lo(i) = nan;
      zx_lo(i) = nan;
    end

    drxlodix(i,icoil) = dx(1);
    dzxlodix(i,icoil) = dx(2);

    % ========
    % upper xp
    % ========
    r = rx_up(i);
    z = zx_up(i);

    validxp = ~isnan(r) &&  (norm([r-rx_up_ref, z-zx_up_ref]) < 0.5);

    if validxp
      psi = reshape(psizr(i,:,:), nz, nr);

      [~, dpsidr, dpsidz] = bicubicHermite(rg, zg, dpsi, r, z);
      [~, ~, ~, psi_rr, psi_zz, psi_rz] = bicubicHermite(rg, zg, psi, r, z);
      J = [psi_rr psi_rz; psi_rz psi_zz];
      dx = -inv(J)*[dpsidr; dpsidz];
    else
      dx = [nan nan];
      rx_up(i) = nan;
      zx_up(i) = nan;
    end

    drxupdix(i,icoil) = dx(1);
    dzxupdix(i,icoil) = dx(2);

  end
end


% save data, set nans to zero
i = isnan(rx_up) | isnan(zx_up) | isnan(rx_lo) | isnan(zx_lo);
rx_up(i) = 0;
zx_up(i) = 0;
rx_lo(i) = 0;
zx_lo(i) = 0;
drxupdix(i,:) = 0;
dzxupdix(i,:) = 0;
drxlodix(i,:) = 0;
dzxlodix(i,:) = 0;

rx_up_filtered = rx_up;
zx_up_filtered = zx_up;
rx_lo_filtered = rx_lo;
zx_lo_filtered = zx_lo;

A = variables2struct(drxlodix, dzxlodix, drxupdix, dzxupdix, rx_up_filtered, zx_up_filtered, rx_lo_filtered, zx_lo_filtered);

if saveit  
%   fds = fieldnames(A);
%   for i = 1:length(fds)
%     A.(fds{i}) = A.(fds{i})(:);
%   end
  
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
plot(gs_drxlodix(isample,icoil))
plot(drxlodix(isample,icoil))
subplot(122)
hold on
plot(gs_dzxlodix(isample,icoil))
plot(dzxlodix(isample,icoil))

figure
subplot(121)
hold on
plot(gs_drxupdix(isample,icoil))
plot(drxupdix(isample,icoil))
subplot(122)
hold on
plot(gs_dzxupdix(isample,icoil))
plot(dzxupdix(isample,icoil))

















































