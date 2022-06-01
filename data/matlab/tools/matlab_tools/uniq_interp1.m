function Vout = uniq_interp1(X, V, Xq)

[~,i] = unique(X);

Vout = interp1(X(i), V(i,:), Xq);
end
