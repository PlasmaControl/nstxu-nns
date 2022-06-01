ccc

% =========
% SETTINGS
% =========
shot = 204660;
time = 100;
coilname = 'PF1AU1';


gfile_dir = '/p/omfit/users/jwai/projects/ramp-up/EFITtime/OUTPUTS/';
shotdir = [gfile_dir num2str(shot) '/gEQDSK'];
eq = read_eq(shot, time/1000, shotdir, 'NSTX');
load('nstxu_obj_config2020_6565.mat')
response = gspert(eq.gdata, tok_data_struct);


struct_to_ws(tok_data_struct);
for i = 1:size(ccnames,1)
  if contains(ccnames(i,:), coilname)
    icoil = i;
    break
  end
end

% convert from grid current response to grid flux response
xpcxpv = mpp_x_vec(tok_data_struct.mpp, response.dcphidis(:,icoil));
xpcxpv = reshape(xpcxpv, nz, nr);

% load('mpc_eqnet.mat')
% mpc = reshape(mpc_eqnet(:,1), nz, nr);
mpc = reshape(mpc(:,icoil), nz, nr);

plot_eq(eq)


figure
plot_nstxu_geo(tok_data_struct)
contour(rg, zg, xpcxpv, 30)
plot_coil_circuit(icoil, tok_data_struct, [1 0 0])
set(gcf, 'Position', [531 30 337 545]);
title(['Plasma-only response to ' coilname])


figure
plot_nstxu_geo(tok_data_struct)
contour(rg, zg, mpc, 30)
plot_coil_circuit(icoil, tok_data_struct, [1 0 0])
set(gcf, 'Position', [531 30 337 545]);
title(['Vacuum response to ' coilname])


figure
plot_nstxu_geo(tok_data_struct)
contour(rg, zg, mpc + xpcxpv, 30)
plot_coil_circuit(icoil, tok_data_struct, [1 0 0])
set(gcf, 'Position', [531 30 337 545]);
title(['Total response to ' coilname])




























