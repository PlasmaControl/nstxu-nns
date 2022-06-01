clear; clc; close all

load('nstxu_obj_config2016_6565.mat')

tok_data_struct.bpsignals = cellstr(tok_data_struct.bpnames);
bpsignals = tok_data_struct.bpsignals;
flsignals = tok_data_struct.flsignals;
lvsignals = tok_data_struct.lvsignals;

iiv = 0;

fid = fopen('nstx_signals.txt');

tline = fgetl(fid);
while ischar(tline)
  
  % check if it matches a bpsignal
  for i = 1:length(bpsignals)    
    if contains(tline, bpsignals{i}, 'IgnoreCase', true)      
      [magnetics.bpsignals.tree{i}, magnetics.bpsignals.tag{i}] = read_tag(tline);      
      break
    end
  end
  
  % check if it matches a flsignal
  for i = 1:length(flsignals)    
    if contains(tline, flsignals{i}, 'IgnoreCase', true)      
      [magnetics.flsignals.tree{i}, magnetics.flsignals.tag{i}] = read_tag(tline);      
      break
    end
  end
  
   % check if it matches a lvsignal
  for i = 1:length(lvsignals)    
    if contains(tline, lvsignals{i}, 'IgnoreCase', true)      
      [magnetics.lvsignals.tree{i}, magnetics.lvsignals.tag{i}] = read_tag(tline);      
      break
    end
  end
  
  % check if it matches a vessel current signal    
  if contains(tline, 'top.pcs.magcontrol.current', 'IgnoreCase', true) 
    iiv = iiv+1;    
    [magnetics.ivsignals.tree{iiv}, magnetics.ivsignals.tag{iiv}] = read_tag(tline);        
  end
    
  tline = fgetl(fid);    
end

save('magnetics_mdsnames.mat', 'magnetics')


% Each line has nice, standardized formatting so we can extract tree and
% tag based on the string patterns
function [tree, tag, omfit_tag] = read_tag(txt)

  s = split(txt);
  omfit_tag = s{1};
  
  tree_tag = split(s{2}, '::TOP');
  
  tree = tree_tag{1}(2:end);
  tag = tree_tag{2};

end




















