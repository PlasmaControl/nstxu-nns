% Did not save the response to li and betap first time through. Rerun
% gspert and append to saved files. 

load('job_args');

targs = load(job_args.save_fn).targs;

addpath(genpath(job_args.MDSPLUS_DIR));
addpath(genpath([job_args.DATA_ROOT '/matlab']));
shot = job_args.shot;

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
    
    targs.dcphidli(i,:) = sys.gspert_data.dcphidli(:);
    targs.dcphidbetap(i,:) = sys.gspert_data.dcphidbetap(:);
    targs.dpsidli(i,:) = mpp*sys.gspert_data.dcphidli(:);
    targs.dpsidbetap(i,:) = mpp*sys.gspert_data.dcphidbetap(:);
    
  catch
    warning('Bad sample!')
    targs.ibad(i) = true;         
  end
end

if job_args.saveit
  save(job_args.save_fn, 'targs', '-v7.3')
end





































































































