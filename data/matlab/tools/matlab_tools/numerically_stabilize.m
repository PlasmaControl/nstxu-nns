% eigmax: maximum distance of an eigenvalue from the origin.
%         i.e. max(abs(Astab)) will be <= eigmax. 

function Astab = numerically_stabilize(A, eigmax)

[V,D] = eig(A);

D = diag(D);

% artificially correct vertical instability (set real part to 0)
k = real(D) > 0;     
D(k) = -eps;

% slow down the super fast stable poles which introduce 
% numerical instabilities
k = abs(D) > eigmax;  
D(k) = D(k) .* eigmax ./ abs(D(k)); 

Astab = V * diag(D) * inv(V);

% corrections can introduce small imaginary components 
Astab = real(Astab);         



















