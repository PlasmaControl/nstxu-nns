% Fetches EFIT01 gfiles for NSTXU, then runs gspert on the equilibria

shot = 203172;
saveit = 0;
save_fn = '';

tree = 'efit01';
tokamak = 'nstxu';
server = 'skylark.pppl.gov:8501';

fprintf('\n\n')
disp(['Fetching shot ' num2str(shot)])
fprintf('\n\n')

eqs = read_eq(shot, 'all', tree, tokamak, server);

tok_data_struct = load('nstxu_obj_config2016_6565.mat').tok_data_struct;
mpp = load('coneqt_nstxu_obj_config2016_6565.mat').tok_data_struct.mpp;

circ = nstxu2016_circ(tok_data_struct);

build_inputs.tokamak = 'NSTXU';
build_inputs.vacuum_objs = tok_data_struct;
build_inputs.ichooseq = 4;
build_inputs.irzresp_dynamic = 5;
build_inputs.irzresp_output = 5;
build_inputs.iplcirc = 1;
build_inputs.cccirc = circ.cccirc(:);
build_inputs.vvcirc = circ.vvcirc(:);
build_inputs.vvgroup = circ.vvgroup(:);

ip_thresh = 1e5;
ip = [eqs.gdata(:).cpasma];
iuse = ip(:) > ip_thresh & eqs.time > 0;

eqs.gdata(~iuse) = [];
eqs.time(~iuse) = [];
times = eqs.time;

coils = fetch_coilcurrents_nstxu(shot, times);

targs = struct;
targs.coil_currents = coils.icx(circ.ikeep,:)';
targs.vessel_currents = coils.ivx';
targs.ibad = false(size(times));

i = 0;
ibad = 0;

for i = 1:length(times)
  
  
  disp(['time_ms: ' num2str(times(i)*1000)])
  
  try
    eq = eqs.gdata(i);
    eq.ecturn = tok_data_struct.ecnturn;
    eq.ecid   = ones(size(eq.ecturn));
    eq.turnfc = tok_data_struct.fcnturn';
    eq.fcturn = circ.fcfrac;
    eq.fcid = circ.fccirc';
    eq = struct_fields_to_double(eq);
        
    build_inputs.equil_data = eq;
    
    sys = build_tokamak_system(build_inputs);
    delete('NSTXU_netlist.dat')
    
    P = circ.Pxx(1:end-1,1:end-1);
    xmatx = sys.xmatx(1:end-1,1:end-1);
    
    xmat = P'*xmatx*P;
    xmat = double(xmat);
    
    e = esort(eig(sys.amat(1:end-1,1:end-1)));
    gamma = real(e(1));
    gamma = double(gamma);
    disp(['gamma: ' num2str(gamma)])
    
    dcphidis = [sys.gspert_data.dcphidis sys.gspert_data.dcphidip(:)];
    dcphidix = dcphidis * circ.Pxx;
    dcphidix = double(dcphidix);
    dpsidix = mpp*dcphidix;
    
    j = ~(eq.rbbbs==0 & eq.zbbbs==0);
    [rbbbs, zbbbs] = interparc(eq.rbbbs(j), eq.zbbbs(j), 20, 0);
    
    % save to struct    
    targs.shot = shot;
    targs.time(i) = times(i);
    targs.actualtime(i) = times(i);    
    targs.xmat(i,:,:) = xmat;
    targs.gamma(i) = gamma;
    targs.dpsidix(i,:,:) = dpsidix;
    targs.ip(i) = eq.cpasma;
    targs.pprime(i,:) = eq.pprime;
    targs.ffprim(i,:) = eq.ffprim;
    targs.pres(i,:) = eq.pres;
    targs.psirz(i,:,:) = eq.psirz;
    targs.pcurrt(i,:,:) = eq.pcurrt;
    targs.rcur(i) = sum(tok_data_struct.rgg(:).*eq.pcurrt(:)) / eq.cpasma;
    targs.zcur(i) = sum(tok_data_struct.zgg(:).*eq.pcurrt(:)) / eq.cpasma;
    targs.rbbbs(i,:) = rbbbs;
    targs.zbbbs(i,:) = zbbbs;
    targs.psibry(i) = eq.psibry;
    targs.psimag(i) = eq.psimag;
    targs.rmaxis(i) = eq.rmaxis;
    targs.zmaxis(i) = eq.zmaxis;
    targs.qpsi(i,:) = eq.qpsi;    
    
  catch
    warning('Bad sample!')
    targs.ibad(i) = true;         
  end
end

if job_args.saveit
  save(job_args.save_fn, 'targs', '-v7.3')
end























































