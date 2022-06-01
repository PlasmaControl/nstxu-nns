% Inputs: eq
function plot_eq(eq)

if isfield(eq,'gdata'), eq = eq.gdata; end

% Load tok_data_struct
load('nstxu_obj_config2016_6565.mat')
rg = tok_data_struct.rg;
zg = tok_data_struct.zg;

plot_nstxu_geo(tok_data_struct)

psizr = eq.psizr;
psibry = eq.psibry;

% time = 1000*eq.time;
% shot = eq.shotnum;

contour(rg,zg,psizr,[psibry psibry],'r', 'linewidth', 3);
contour(rg,zg,psizr,30,'b', 'linewidth', 0.5);

% set(gcf,'Position',[204 38 312 533])

% title([num2str(shot) ': ' num2str(time) 'ms'])















