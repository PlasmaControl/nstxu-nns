
function [rx1, zx1, rx2, zx2] = gspert_xp_finder(eq)

struct_to_ws(eq);

% FIND O-POINTS, X-POINTS

% index all potentially important points
i = islocalmin(psizr, 'FlatSelection', 'all') | islocalmax(psizr, 'FlatSelection', 'all');
j = islocalmin(psizr,2, 'FlatSelection', 'all') | islocalmax(psizr,2, 'FlatSelection', 'all');
i = i | circshift(i,1) | circshift(i,-1) | circshift(i',1)' | circshift(i',-1)';
j = j | circshift(j,1) | circshift(j,-1) | circshift(j',1)' | circshift(j',-1)';
k = find(i & j);


% in = find(inpolygon(rgg(k), zgg(k), rlim, zlim));
in = 1:length(k);


[rgg,zgg] = meshgrid(rg,zg);
rsearch = rgg(k(in));
zsearch = zgg(k(in));


% zoom in on x/o-points
for i = 1:length(in)
  try
    [rxo(i), zxo(i), psixo(i)] = isoflux_xpFinder(psizr, rsearch(i), zsearch(i), rg, zg);
  catch
    rxo(i) = inf;
    zxo(i) = inf;
    psixo(i) = inf;
  end
end

% choose points that are within the valid grid
in = rxo > rg(3) & rxo < rg(end-2) & zxo > zg(3) & zxo < zg(end-2);
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


% x-points found are valid. Now identify the dominant x-point above axis,
% and dominant x-point below axis, by comparing to boundary
i = rbbbs ~= 0 & zbbbs ~= 0;
rbbbs = rbbbs(i);
zbbbs = zbbbs(i);

[~, psi_r, psi_z] = bicubicHermite(rg, zg, psizr, rbbbs, zbbbs);
[~,i] = min(psi_r.^2 + psi_z.^2);

r = rbbbs(i); % bry point closest to dominant x-point
z = zbbbs(i);

[~,j] = min((rx-r).^2 + (zx-z).^2);
rx1 = rx(j);
zx1 = zx(j);

% dominant x-point on other side of axis
i = sign(zbbbs) ~= sign(zx1);
rbbbs = rbbbs(i);
zbbbs = zbbbs(i);

[~, psi_r, psi_z] = bicubicHermite(rg, zg, psizr, rbbbs, zbbbs);
[~,i] = min(psi_r.^2 + psi_z.^2);

r = rbbbs(i); % bry point closest to dominant x-point
z = zbbbs(i);

[~,j] = min((rx-r).^2 + (zx-z).^2);
rx2 = rx(j);
zx2 = zx(j);

















