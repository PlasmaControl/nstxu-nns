ccc
load('/Users/jwai/Research/rampup_nstxu/dev/sim_inputs204660.mat')
% load('/Users/jwai/Research/rampup_nstxu/sim/sim_inputs204660.mat')
load('nstxu_obj_config2016_6565.mat')

iv = sim_inputs.traj.iv;
iv_pred = sim_inputs.x_all(9:48,:)';
[vvgroup,~,vvnames] = nstxu2020_vvcirc;
vvdata = tok_data_struct.vvdata;
z   = vvdata(1,:);
r   = vvdata(2,:);
dz  = vvdata(3,:);
dr  = vvdata(4,:);
ac  = vvdata(5,:);
ac2 = vvdata(6,:);
options.iaxes  = 1;
options.icolor = 0;
options.igrid  = 0;
options.ifl    = 0;
options.ifull  = 0;



figure
for i = 1:length(vvnames)
  i
  clf
  
  subplot(1,2,2)
  hold on
%   plot(tok_data_struct.limdata(2,:), tok_data_struct.limdata(1,:), 'k')
  plot_nstxu_geo(tok_data_struct, options)
  axis([0.2 1.8 -2 2])
  axis equal
  for j = find(vvgroup == i)
    plot_efit_region(z(j), r(j), dz(j), dr(j), ac(j), ac2(j), [1 0 0]);
  end
  scatter(r(j), z(j), 100, 'r', 'filled')
   title(vvnames(i,:), 'fontsize', 14)
  
  subplot(1,2,1)
  hold on
  plot(iv, 'color', [1 1 1] * 0.8)
  plot(iv(:,i), '--r', 'linewidth', 2)
  plot(iv_pred(:,i), 'b', 'linewidth', 2)
  title(vvnames(i,:), 'fontsize', 14)
  set(gcf,'Position',[6 151 722 349])
end

  















