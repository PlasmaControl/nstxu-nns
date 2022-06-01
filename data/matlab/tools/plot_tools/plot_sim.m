ccc
load('/Users/jwai/Research/rampup_nstxu/sim/matlab.mat')

t = 70;
load(['eq204660_' num2str(t) '.mat'])
load('nstxu_obj_config2020_6565.mat')

% plot_nstxu_geo(tok_data_struct)
[~,cs] = contour(eq.rg, eq.zg, eq.psizr, 30, '--r');
hold on

[~,i] = min(abs(t - tspan*1000));
tspan(i)*1000

contour(eq.rg, eq.zg, squeeze(sim.psizr(i,:,:)), cs.LevelList, '-b')


set(gcf, 'Position', [583 105 426 684])

legend('Experiment', 'Simulation', 'fontsize', 16)
title(['204660 @ ' num2str(t) 'ms'], 'fontsize', 18)







