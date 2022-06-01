% Extract several mutual inductances and resistances, apply circuit
% connections, and save in format easy for the python neural nets to
% access.

saveit = 0;

tok_data_struct = load('nstxu_obj_config2016_6565.mat').tok_data_struct;
Mpp = load('coneqt_nstxu_obj_config2016_6565.mat').tok_data_struct.mpp;

Mppi = inv(Mpp);

circ = nstxu2016_circ(tok_data_struct);

Pxx = circ.Pxx(1:end-1, 1:end-1);

R = [tok_data_struct.resc; tok_data_struct.resv];
Rxx = diag(Pxx' * diag(R) * Pxx);
Rxx(circ.iremove) = 10;
Rxx = diag(Rxx);

MpcMpv = [tok_data_struct.mpc tok_data_struct.mpv];

M = [tok_data_struct.mcc tok_data_struct.mcv; tok_data_struct.mcv' tok_data_struct.mvv];
M = Pxx' * M * Pxx;

mpcx = tok_data_struct.mpc * circ.Pcc * circ.Pcc_keep';
mpvx = tok_data_struct.mpv * circ.Pvv; 


if saveit
  fn = {'M', 'MpcMpv', 'Rxx', 'Pxx', 'Mpp', 'Mppi', 'mpcx', 'mpvx'};
  for i = 1:length(fn)
    save(fn{i}, fn{i})
  end
end







