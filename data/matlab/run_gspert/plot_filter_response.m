% Smooths the gspert response data and applies a variety of threshold
% filters to determine whether or not to keep the data point. Criteria are:
%
% Don't use sample point if sample does not meet set of criteria:
% 1. growth rate (gamma) is not changing too rapidly (gamma - smooth(gamma)
% < 20Hz)
% 2. growth rate not excessively large (<300 Hz)
% 3. Ip sufficiently large (> 0.1 MA)
%
% Don't use shot at all if shot does not meet criteria:
% 1. shot length > 300ms
% 2. At least 70% of samples meet the sample criteria above

clear; clc; close all;

saveit = 1;
plotit = 0;
WINDOW = 5;

mode = 'val';

DATA_ROOT = getenv('DATA_ROOT');
build_dir = [DATA_ROOT '/rawdata/data_by_shot/' mode '/'];

d = dir(build_dir);
d(1:2) = [];

% geo files for evaluating growth rate
obj_dir = [DATA_ROOT '/matlab/run_gspert/obj/'];
M = load([obj_dir 'M.mat']).M;
Mpp = load([obj_dir 'Mpp.mat']).Mpp;
MpcMpv = load([obj_dir 'MpcMpv.mat']).MpcMpv;
Pxx = load([obj_dir 'Pxx.mat']).Pxx;
Rxx = load([obj_dir 'Rxx.mat']).Rxx;
Mppi = inv(Mpp);

good_shots = [];
bad_shots = [];


% smoothing + plots
for ishot = 1:length(d)
  
  disp(ishot / length(d))
 
  fn = [d(ishot).folder '/' d(ishot).name];
  
  targs = load(fn).targs;
  t = targs.time;
  nsamples = length(t);
  
  % Don't use sample point if sample does not meet set of criteria:
  % 1. growth rate (gamma) is not changing too rapidly (gamma - smooth(gamma)
  % < 20Hz)
  % 2. growth rate not excessively large (<300 Hz)
  % 3. Ip sufficiently large (> 0.1 MA)
  
  gamma_smooth = smoothdata(targs.gamma, 'movmedian', 15);
  e_gamma = abs(targs.gamma - gamma_smooth);
  igood = targs.gamma < 300 & e_gamma < 20 & targs.ip > 1e5;
  
  % Don't use shot at all if shot does not meet criteria:
  % 1. shot length > 300ms
  % 2. At least 70% of samples meet the sample criteria above
  
  shot_is_good = max(t) > 0.3 & sum(igood)/length(igood) > 0.7;
  
  y = targs.dpsidix;
  ysm = smoothdata(y(igood,:,:), 'movmedian', WINDOW);
  ysm = interp1(t(igood), ysm, t, 'linear', 'extrap');
    
  gamma_pred_smooth = t*0;
  
  for i = 1:nsamples
    yi = squeeze(ysm(i,:,1:end-1));
    X = Pxx' * MpcMpv' * Mppi * yi;
    amat = -inv(M+X)*Rxx;
    gamma_pred_smooth(i) = max(real(eig(amat)));
  end
  
  % ensure smoothing data did not change gamma too much. If it changed it
  % by >20hz then dont include sample.
  e_gamma = abs(gamma_smooth - gamma_pred_smooth);
  igood2 = e_gamma < 20; 
  igood = igood & igood2;
  
  targs = append2struct(targs, igood, shot_is_good, gamma_pred_smooth);    
  targs.('dpsidix_smooth') = ysm;
  
  if shot_is_good
    good_shots(end+1) = targs.shot;
  else
    bad_shots(end+1) = targs.shot;
  end
  
  if saveit
    save(fn, 'targs')
  end
  
  if plotit
    for icoil = [1]
      figure
      hold on
      sgtitle(['Shot ' num2str(targs.shot) ' Coil ' num2str(icoil)], 'fontsize', 14)
      set(gcf, 'Position', [1085 86 560 642])
      ax(1) = subplot(311);
      hold on
      grid on
      plot(t, targs.gamma, 'linewidth', 1.5)
      plot(t(igood), gamma_pred_smooth(igood), 'linewidth', 1.5, 'linestyle', '--')
      % scatter(t, targs.gamma)
      ylim([-10 min(max(targs.gamma), 200)])
      
      ax(2) = subplot(312);
      hold on
      grid on
      plot(t, reshape(y(:,:,icoil), nsamples, []));
      
      ax(3) = subplot(313);
      hold on
      grid on
      plot(t, reshape(ysm(:,:,icoil), nsamples, []));
      ylimits = ylim;
      
      linkaxes(ax, 'x')
      linkaxes(ax(2:3), 'y')
      ylim(ylimits)
      drawnow
    end
  end
  
  
%   figure
%   plot(reshape(targs.dpsidix_smooth(:,:,icoil), nsamples, []))
%   ylim(ylimits)
%   hold on
%   grid on
%   set(gcf, 'Position', [501 130 566 172])
%   drawnow
  
end


disp([ num2str(length(good_shots)) ' good shots:'])
disp(good_shots)

disp([ num2str(length(bad_shots)) ' bad shots:'])
disp(bad_shots)





















































