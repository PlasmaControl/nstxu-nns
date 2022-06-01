% inputs shot, time(ms)
shot = 204660;
time = 240;

load('/u/jwai/ramp_up_nstxu/mat_files/nstxu_obj_config2016_6565.mat')
plot_nstxu_geo(tok_data_struct)
fpath = '/u/jwai/rampup_nstxu2/eq/geqdsk/';

[eq, neq] = read_eq(shot, time/1000, fpath);

rg = tok_data_struct.rg;
zg = tok_data_struct.zg;
psizr = eq.gdata.psizr;
psibry = eq.gdata.psibry;

contour(rg,zg,psizr,[psibry psibry],'b');
title([num2str(shot) ': ' num2str(time) 'ms'])















