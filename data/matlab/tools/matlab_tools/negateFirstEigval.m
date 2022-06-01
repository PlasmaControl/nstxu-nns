function Mhat = negateFirstEigval(M)

[V,D] = eig(M);
eig_max = max(real(diag((D))));

if eig_max > 0
    [row, col] = find(real(D) == eig_max);
    D(row,col) = -D(row,col);
    M = V * D / V;
end

Mhat = M;

end
