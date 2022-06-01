% From the equilibrium flux psizr and coil+vessel currents, finds the
% plasma flux psizr_pla. psizr = psizr_app + psizr_pla, where psizr_app =
% Mpc*ic + Mpv*iv (mutual inductances * coil and vessel currents).

clear all; clc; close all

plotit = 1;
saveit = 0;
mode = 'all_eqnet';

DATA_ROOT = getenv('DATA_ROOT');
datadir = [DATA_ROOT '/rawdata/data_by_var/' mode '/'];

load('nstxu_obj_config2016_6565.mat')
circ = nstxu2016_circ(tok_data_struct);

mpc = tok_data_struct.mpc * circ.Pcc * circ.Pcc_keep';
mpv = tok_data_struct.mpv * circ.Pvv;

ic = load([datadir 'coil_currents.mat']).coil_currents;
iv = load([datadir 'vessel_currents.mat']).vessel_currents;

psizr_app = (ic*mpc' + iv*mpv') / (-2*pi); % -2pi scaling factor due to toksys vs Efit formats
psizr_app_ic = ic*mpc' / (-2*pi);   % applied flux from only the PF coils (vessel currents are not known as accurately)

psizr_app = reshape(psizr_app, [], 65, 65);
psizr_app_ic = reshape(psizr_app_ic, [], 65, 65);

psirz = load([datadir 'psirz.mat']).psirz;
psizr = permute(psirz, [1 3 2]);

psizr_pla = psizr - psizr_app;
psizr_pla_iv = psizr-psizr_app_ic;

if saveit
  save([datadir 'psizr_pla.mat'], 'psizr_pla', '-v7.3'); 
  save([datadir 'psizr_pla_iv.mat'], 'psizr_pla_iv', '-v7.3'); 
end

if plotit
  i = 10;
  figure
  contour(squeeze(psizr_app(i,:,:)), 40);
  colorbar
  
  figure
  contour(squeeze(psizr_pla(i,:,:)), 40);
  colorbar
end


















