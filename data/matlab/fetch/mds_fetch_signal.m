% use times=[] to fetch all available times

function signal = mds_fetch_signal(shot, tree, times, tag, plotit, smoothit)

if ~exist('plotit','var'), plotit = 0; end
if ~exist('smoothit','var'), smoothit = 0; end

mdshost = 'skylark.pppl.gov:8501';
mdsconnect(mdshost);
mdsopen(tree, shot);
mdstimes = mdsvalue(strcat('dim_of(', tag, ')'));
mdssigs = mdsvalue(tag);


if isempty(times)
  times = mdstimes;
  sigs = mdssigs;
else
  
  if ischar(mdssigs)
    warning('No data')
  end
  
  if length(mdstimes) ~= size(mdssigs,1)
    mdssigs = mdssigs';
  end
  
  if smoothit
    dt = mean(diff(times));
    dt_mds = mean(diff(mdstimes));    
    window = floor(dt / dt_mds);
    mdssigs_smooth = smoothdata(mdssigs, 'movmean', window);
    sigs = interp1(mdstimes, mdssigs_smooth, times)';
  else
     sigs = interp1(mdstimes, mdssigs, times)';
  end
  
 
  
end


signal = variables2struct(shot,tag,times,sigs);

% mdsclose;
% mdsdisconnect;

if plotit
  % figure
  hold on  
  plot(mdstimes, mdssigs, 'linewidth', 2)
  % plot(times, sigs, 'linewidth', 1)
  
  if length(times) <= 10
    for i = 1:length(times)
      xline(times(i), '-k', 'linewidth', 1.5);
    end
  end
end