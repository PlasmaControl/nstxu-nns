function [u,s,v] = svd_tol(A, energy_thresh)


[u,s,v] = svd(A, 'econ');

% only save n_components
if exist('energy_thresh', 'var') && energy_thresh < 1
  energies = cumsum(diag(s)) / sum(diag(s));
  n_components = find(energies >= energy_thresh, 1);
  
  u = u(:, 1:n_components);
  s = s(1:n_components, 1:n_components);
  v = v(:, 1:n_components);
end








