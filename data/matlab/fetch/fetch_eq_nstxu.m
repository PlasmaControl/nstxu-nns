% Import an equilibrium for nstxu from the mdsplus tree
%
% This builds off the function read_eq.m included with Toksys, but also fetches
% the coil current and modifies a few fields so that the coil current 
% vectors are consistent with geometry. 
%
% Josiah Wai, 3/1/2021

% EXAMPLE:
% shot = 204660;
% time = 0.200;
% tree = 'EFIT01';
% tokamak = 'nstxu';
% server = 'skylark.pppl.gov:8501';
% opts.cache_it = 0;
% opts.cache_dir = '/Users/jwai/Research/rampup_nstxu/eq/geqdsk_import2';
% opts.plotit = 1;
% eq = fetch_eq_nstxu(shot, time, tree, tokamak, server, opts)


function eq = fetch_eq_nstxu(shot, time, tree, tokamak, server, opts)

if ~exist('tree', 'var'), tree = 'EFIT01'; end
if ~exist('tokamak', 'var'), tokamak = 'nstxu'; end
if ~exist('server', 'var'), server = 'skylark.pppl.gov:8501'; end
if ~exist('opts','var'), opts = struct; end
if ~isfield(opts, 'cache_it'), opts.cache_it = 0; end
if ~isfield(opts, 'plotit'), opts.plotit = 0; end   
if ~isfield(opts, 'cache_dir'), opts.cache_dir = nan; end   
if ~isfield(opts, 'force_mds_load'), opts.force_mds_load = 0; end

% load from cache
if ~opts.force_mds_load
  try    
    save_fn = [opts.cache_dir '/eq' num2str(shot) '_' num2str(floor(time*1000)) '.mat'];
    eq = load(save_fn).eq;
    load_from_mds = 0;
    disp('Equilibrium was loaded from cache. (To force new load, ')
    disp('set opts.force_mds_load = true.)')     
  catch
    load_from_mds = 1;
  end
end

if load_from_mds      
  
  eq = read_eq(shot,time,tree,tokamak,server);

  % The coil current vector eq.cc is confusing and dependent on geometry in
  % arcane ways. Instead of inverting eq.cc to eq.ic, we will load ic directly
  % from mds, define our geometry here, and calculate what eq.cc is for
  % our geometry.  

  tok_data_struct = load('nstxu_obj_config2016_6565.mat').tok_data_struct;
  circ = nstxu2016_circ(tok_data_struct);

  signal = mds_fetch_signal(shot, tree, time, '.RESULTS.AEQDSK:ECCURT', opts.plotit);  % OH
  ecefit = signal.sigs;

  signal = mds_fetch_signal(shot, tree, time, '.RESULTS.AEQDSK:CCBRSP', opts.plotit); % PF coils + vessel 
  ccefit = signal.sigs;

  icx = zeros(13,1);
  ivx = zeros(40,1);

  icx(1) = ecefit;    % OH
  icx(2) = ccefit(1); % PF1AU
  icx(3) = ccefit(2); % PF1BU
  icx(4) = ccefit(3); % PF1CU
  icx(5) = ccefit(4); % PF2U
  icx(6) = ccefit(5); % PF3U
  icx(7) = (ccefit(6)+ccefit(9))/2.0;  % PF4
  icx(8) = (ccefit(7)+ccefit(8))/2.0;  % PF5
  icx(9) = ccefit(10);  % PF3L
  icx(10) = ccefit(11); % PF2L
  icx(11) = ccefit(12); % PF1CL
  icx(12) = ccefit(13); % PF1BL
  icx(13) = ccefit(14); % PF1AL

  ivx(:) = ccefit(15:end);

  ic = circ.Pcc * icx;
  iv = circ.Pvv * ivx;

  % copy circuit information into eq
  eq.ecturn = tok_data_struct.ecnturn;
  eq.ecid   = ones(size(eq.ecturn));

  eq.turnfc = tok_data_struct.fcnturn';
  eq.fcturn = circ.fcfrac;
  eq.fcid = circ.fccirc';

  % Convert from ic to cc:
  % The ic vector already represents coil currents in toksys format. Use a
  % hack to convert from ic to efit format, so that gs codes can convert
  % back from efit format to toksys.
  idiot = eq;
  ncc = length(ic);
  for j = 1:ncc
    idiot.cc = zeros(ncc,1);
    idiot.cc(j) = 1;
    equil_I = cc_efit_to_tok(tok_data_struct,idiot);
    iccc(:,j) = equil_I.cc0t;
  end
  piccc = pinv(iccc);
  cc = piccc * ic;

  eq = append2struct(eq, cc, ic, iv, icx, ivx);

  % error check - convert cc back to ic
  equil_I = cc_efit_to_tok(tok_data_struct, eq);
  cc = equil_I.cc0t;

  if max(abs(cc - ic)) > sqrt(eps)
    error('Something is wrong with coil current vectors.')
  end
  
  if opts.cache_it  
    save(save_fn, 'eq')
  end
  
end

% plot it
if opts.plotit
  figure
  plot_eq(eq)
end


% store everything in double precision
fn = fieldnames(eq);
for i = 1:length(fn)
  if isnumeric(eq.(fn{i}))
    eq.(fn{i}) = double(eq.(fn{i}));
  end
end

























