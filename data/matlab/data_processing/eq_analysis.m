
function shp = eq_analysis(psizr, psibry, rbbbs, zbbbs, tok_data_struct, mppi, plotit)

if ~exist('plotit', 'var'), plotit = 0; end

xlim = tok_data_struct.limdata(2,:);
ylim = tok_data_struct.limdata(1,:);
rg = tok_data_struct.rg;
zg = tok_data_struct.zg;
rgg = tok_data_struct.rgg;
zgg = tok_data_struct.zgg;

% limited or diverted
[rzbbbs_fine] = contourc(rg,zg,psizr,[psibry psibry]);
rbbbs_fine = rzbbbs_fine(1,:);
zbbbs_fine = rzbbbs_fine(2,:);

e = 0.002;

i = rbbbs_fine < min(rbbbs) - e | rbbbs_fine > max(rbbbs) + e | ...
    zbbbs_fine < min(zbbbs) - e | zbbbs_fine > max(zbbbs) + e;

rbbbs_fine(i) = [];
zbbbs_fine(i) = [];

[~,dist] = distance2curve([xlim(:) ylim(:)], [rbbbs_fine(:) zbbbs_fine(:)]);
  
islimited = double(min(dist) < .005);

if isempty(islimited), islimited = nan; end
  
% x-points
try
    
  % find best upper and lower guesses for x-point
  iup = zbbbs > (max(zbbbs) + min(zbbbs))/2;
  ilo = find(~iup);
  iup = find(iup);
  [~,psi_r,psi_z] = bicubicHermite(rg,zg,psizr,rbbbs,zbbbs);

  [~,i] = min( psi_r(iup).^2 + psi_z(iup).^2);
  [~,j] = min( psi_r(ilo).^2 + psi_z(ilo).^2);  

  rx_up = rbbbs(iup(i));
  zx_up = zbbbs(iup(i));

  rx_lo = rbbbs(ilo(j));
  zx_lo = zbbbs(ilo(j));
  
  % zoom in on x-point
  [rx_up,zx_up] = isoflux_xpFinder(psizr,rx_up,zx_up,rg,zg);
  [rx_lo,zx_lo] = isoflux_xpFinder(psizr,rx_lo,zx_lo,rg,zg);

  % check if inside limiter
  if ~all(inpolygon([rx_lo rx_up], [zx_lo zx_up], xlim, ylim))
    rx_lo = nan;
    zx_lo = nan;
    rx_up = nan;
    zx_up = nan;
  end
  
catch
  rx_lo = nan;
  zx_lo = nan;
  rx_up = nan;
  zx_up = nan;
end


% current centroid
currt = mppi * psizr(:);
mask = inpolygon(rgg, zgg, rbbbs_fine, zbbbs_fine);
pcurrt = currt;
pcurrt(~mask) = 0;
rcur = sum(pcurrt.*rgg(:)) / sum(pcurrt(:));
zcur = sum(pcurrt.*zgg(:)) / sum(pcurrt(:));


% shape control gaps
rseg1 = [1.2 0.75];
zseg1 = [-1.2083 -0.3];
rseg2 = rseg1;
zseg2 = -zseg1;

sp1 = isoflux_spFinder(psizr, psibry, rg, zg, [zseg1; rseg1], 1:2);
sp2 = isoflux_spFinder(psizr, psibry, rg, zg, [zseg2; rseg2], 1:2);
sp1 = empty2nan(sp1);
sp2 = empty2nan(sp2);

gap1 = sqrt((rseg1(1) - sp1(1))^2 + (zseg1(1) - sp1(2))^2);
gap2 = sqrt((rseg2(1) - sp2(1))^2 + (zseg2(1) - sp2(2))^2);



% elongation, triangularity, Romp
rmax = max(rbbbs);
rmin = min(rbbbs);
[zmax,i] = max(zbbbs);
rup = rbbbs(i);
[zmin,i] = min(zbbbs);
rlow = rbbbs(i);
b = (zmax-zmin)/2;
a = (rmax-rmin)/2;
kappa = b/a;
R0 = rmin+a;
delta = (2*R0 - rup - rlow) / (2*a);
Romp = rmax;


if plotit
  figure
  plot_nstxu_geo(tok_data_struct)
  % plot_eq(eq)
  scatter(sp1(1), sp1(2), 100, 'k', 'filled')
  scatter(sp2(1), sp2(2), 100, 'k', 'filled')
  plot(rseg1, zseg1, 'b', 'linewidth', 2)
  plot(rseg2, zseg2, 'b', 'linewidth', 2)
  scatter(rcur, zcur, 100, 'filled')
  scatter(rx_lo, zx_lo, 100, 'rx')
  scatter(rx_up, zx_up, 100, 'rx')  
  contour(rg,zg,psizr,[psibry psibry])
  drawnow
end

shp = variables2struct(islimited, rx_lo, zx_lo, rx_up, zx_up, rcur, zcur, ...
  gap1, gap2, a, b, kappa, delta, R0, rmax);














