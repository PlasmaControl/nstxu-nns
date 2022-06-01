ccc
A = load('/Users/jwai/Research/rampup_nstxu/buildmodel/built_models/std/204660_700_sys.mat').sys.A;

[V,D] = eig(A);

D = diag(D);

% artificially correct vertical instability
k = real(D) > 0;     
D(k) = -D(k);

% slow down the super fast stable poles which introduce 
% numerical instabilities
k = abs(D) > 1e5;  
D(k) = D(k) .* 1e5 ./ abs(D(k)); 

Astab = V * diag(D) * inv(V);

% corrections can introduce small imaginary components 
Astab = real(Astab);         



















