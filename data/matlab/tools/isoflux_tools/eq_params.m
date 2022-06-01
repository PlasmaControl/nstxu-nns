% time = 200;
% load(['/Users/jwai/Research/rampup_nstxu/eq/geqdsk_import/eq204660_' num2str(time) '.mat'])
% psizr = eq.psizr;
% load('nstxu_obj_config2016_6565.mat')
% geom = eq_params(psizr, tok_data_struct, true)

function geom = eq_params(psizr, tok_data_struct, plotit)

if ~exist('plotit', 'var'), plotit=1; end

struct_to_ws(tok_data_struct);
rlim = limdata(2,:);
zlim = limdata(1,:);

[rlim, zlim] = interparc(rlim, zlim, 500, true, true);
rlim(end:end+3) = rlim(1:4); % wrap a few values, helps later with finding local extrema
zlim(end:end+3) = zlim(1:4);


% PLOT
if plotit
  figure(1)
  clf
  hold on
  grid on
  plot(rlim, zlim, 'k', 'linewidth', 2)
  contour(rg, zg, psizr, 100, 'color', [1 1 1]*0.8)
  set(gcf, 'Position', [2110 410 379 650])
  axis equal
end

% FIND O-POINTS, X-POINTS

% index all potentially important points
i = islocalmin(psizr, 'FlatSelection', 'all') | islocalmax(psizr, 'FlatSelection', 'all');
j = islocalmin(psizr,2, 'FlatSelection', 'all') | islocalmax(psizr,2, 'FlatSelection', 'all');
i = i | circshift(i,1) | circshift(i,-1) | circshift(i',1)' | circshift(i',-1)';
j = j | circshift(j,1) | circshift(j,-1) | circshift(j',1)' | circshift(j',-1)';
k = find(i & j);

in = find(inpolygon(rgg(k), zgg(k), rlim, zlim));

rsearch = rgg(k(in));
zsearch = zgg(k(in));

% zoom in on x/o-points
for i = 1:length(in)
  [rxo(i), zxo(i), psixo(i)] = isoflux_xpFinder(psizr, rsearch(i), zsearch(i), rg, zg);
end

% choose points inside limiter
in = find(inpolygon(rxo, zxo, rlim, zlim));
rxo = rxo(in);
zxo = zxo(in);
psixo = psixo(in);

% find unique points to within 1cm
tol = 0.01;
[~,~,idx_r] = uniquetol(rxo, tol);
[~,~,idx_z] = uniquetol(zxo, tol);
[~,i] = unique(idx_r);
[~,j] = unique(idx_z);
k = unique(union(i,j));
rxo = rxo(k);
zxo = zxo(k);
psixo = psixo(k);

% sort whether x-point or o-point
[psi, psi_r, psi_z, psi_rr, psi_zz] = bicubicHermite(rg,zg,psizr,rxo,zxo);
is_xpt = sign(psi_rr) ~= sign(psi_zz);

ro = rxo(~is_xpt);
zo = zxo(~is_xpt);
psio = psixo(~is_xpt);

rx = rxo(is_xpt);
zx = zxo(is_xpt);
psix = psixo(is_xpt);

% check psi inside and outside limiter to see whether magnetic axis should
% be a local min or max
in = inpolygon(rgg, zgg, rlim, zlim);
psi_in = median(psizr(in));
psi_out = median(psizr(~in));
local_sign_maxis = sign(psi_in-psi_out);

% select magnetic axis from among o-points
[~,i] = max(local_sign_maxis*psio);
rmaxis = ro(i);
zmaxis = zo(i);
psimag = psio(i);


% FIND LOCAL EXTREMA ALONG LIMITER
psilim = bicubicHermite(rg,zg,psizr,rlim,zlim);
i = islocalmax(local_sign_maxis*psilim, 'FlatSelection', 'all');
rtouch = rlim(i);
ztouch = zlim(i);


% TRACE BOUNDARIES TO FIND WHICH CANDIDATES ARE CLOSED CONTOURS

% rationale: the true boundary-defining point must be an x-point interior to
% limiter, or a local flux extrema (along the limiter). From all these
% candidate points, the bdef-pt is the one that forms the largest closed contour
rcandidates = [rx(:); rtouch(:)];
zcandidates = [zx(:); ztouch(:)];
[rbdef, zbdef] = trace_contour(rg,zg,psizr,rcandidates,zcandidates,rmaxis,zmaxis,rlim,zlim,0,1);

% All of the remaing (rbdef, zbdef) now form valid closed contours. 
% Choose the most external contour
psibdef = bicubicHermite(rg,zg,psizr,rbdef,zbdef);
[~,i] = min(local_sign_maxis*psibdef);
rbdef = rbdef(i);
zbdef = zbdef(i);
psibry = psibdef(i);

[~,~,~,rbbbs,zbbbs] = trace_contour(rg,zg,psizr,rbdef,zbdef,rmaxis,zmaxis,rlim,zlim,plotit,1);
rbbbs = rbbbs{1};
zbbbs = zbbbs{1};

% sort between upper and lower x-points
[~,i] = sort(abs(psix-psibry)); % first distinguish importance by closeness to boundary
rx = rx(i);
zx = zx(i);
psix = psix(i); 
j = find(zx < zmaxis, 1); % find most important lower xp 
rx1 = rx(j);
zx1 = zx(j);
psix1 = psix(j);
j = find(zx > zmaxis, 1); % find most important upper xp 
rx2 = rx(j);
zx2 = zx(j);
psix2 = psix(j);

if plotit
  scatter(rmaxis, zmaxis, 100, 'filled')
  scatter(rbdef, zbdef, 100, 'm', 'filled')
  plot(rx, zx, 'xm', 'linewidth', 4, 'markersize', 12)
end


% current centroid
if exist('mppi', 'var')
  currt = mppi * psizr(:);
  mask = inpolygon(rgg, zgg, rbbbs, zbbbs);
  pcurrt = currt;
  pcurrt(~mask) = 0;
  rcur = sum(pcurrt.*rgg(:)) / sum(pcurrt(:));
  zcur = sum(pcurrt.*zgg(:)) / sum(pcurrt(:));
else
  rcur = nan;
  zcur = nan;
end


% find major radius R0 and minor radius a and some geometric params
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

rx_lo = rx1;
zx_lo = zx1;
rx_up = rx2;
zx_up = zx2;

% measure gaps
rseg1 = [1.2 0.75];
zseg1 = [-1.2083 -0.3];
rseg2 = rseg1;
zseg2 = -zseg1;


% isoflux_spFinder
sp1 = isoflux_spFinder(psizr, psibry, rg, zg, [zseg1; rseg1], 1:2);
sp2 = isoflux_spFinder(psizr, psibry, rg, zg, [zseg2; rseg2], 1:2);
sp1 = empty2nan(sp1);
sp2 = empty2nan(sp2);

gap1 = sqrt((rseg1(1) - sp1(1))^2 + (zseg1(1) - sp1(2))^2);
gap2 = sqrt((rseg2(1) - sp2(1))^2 + (zseg2(1) - sp2(2))^2);

if plotit
  scatter(sp1(1), sp1(2), 100, 'k', 'filled')
  scatter(sp2(1), sp2(2), 100, 'k', 'filled')
  plot(rseg1, zseg1, 'b', 'linewidth', 2)
  plot(rseg2, zseg2, 'b', 'linewidth', 2)
  drawnow
end

% downsample boundary
i = floor(linspace(1, length(rbbbs), 50));
rbbbs = rbbbs(i);
zbbbs = zbbbs(i);

geom = variables2struct(psibry, rbdef, zbdef, rx1, zx1, psix1, rx2, zx2, psix2, ...
  rbbbs, zbbbs, rmaxis, zmaxis, psimag, kappa, a, b, R0, Romp, delta, gap1, gap2, ...
  rmax, rx_lo, zx_lo, rx_up, zx_up, rcur, zcur);

geom = empty2nan(geom);

end























