% EXAMPLE: 

% labels = {'test1', 'test2'}
% linestyle = '-'
% linewidth = {0.5, 2}
% color = {[1 0 0], [0 0 1]}
% marker = 'none'
% location = 'northwest'
% plot(1:10, rand(10,1)); hold on; plot(1:10, rand(10,1));
% mylegend(labels,linestyle,linewidth,color,marker,location)

function l = mylegend(labels, linestyle, linewidth, color, marker, location, fontsize)

N = length(labels);
ha = zeros(N,1);

% set default values
if ~exist('linestyle', 'var') || isempty(linestyle), linestyle = '-'; end
if ~exist('linewidth', 'var') || isempty(linewidth), linewidth = 1.5; end
if ~exist('color', 'var')     || isempty(color),     color = [0 0 0]; end
if ~exist('marker', 'var')    || isempty(marker),    marker = 'none'; end
if ~exist('location', 'var')  || isempty(location),  location = 'northwest'; end
if ~exist('fontsize','var')   || isempty(fontsize),  fontsize = 7.5     ; end

% repeat args if necessary
if length(linestyle) == 1, linestyle = repcell(linestyle,N); end
if length(linewidth) == 1, linewidth = repcell(linewidth,N); end
if size(color,2)     ~= N, color     = repcell(color,N);     end
if min(size(marker)) == 1, marker    = repcell(marker,N);    end


% plot and legend
for k = 1:N
    ha(k) = plot(NaN,NaN, 'linestyle', linestyle{k}, 'linewidth', ...
      linewidth{k}, 'color', color{k}, 'marker', marker{k});
end

l = legend(ha, labels, 'location', location, 'fontsize', fontsize); % , 'fontweight', 'bold', 'fontsize', fontsize);


function c = repcell(val,n)
  c = cell(n,1);
  c(:) = {val};
end
end








