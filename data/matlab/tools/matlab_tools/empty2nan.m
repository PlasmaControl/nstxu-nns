% empty2nan conversion for vectors and structs of vectors.
% If vector is empty, convert it to nan of equivalent size. 

function X = empty2nan(X)
  
  % vector empty2nan
  if isnumeric(X) && isempty(X)
    sz = size(X);
    sz(sz==0) = 1;
    X = nan(sz);
  end
  
  % struct empty2nan
  if isstruct(X)
    fns = fieldnames(X);
    for i = 1:length(fns)
      Y = X.(fns{i});
      if isnumeric(Y) && isempty(Y)
        sz = size(Y);
        sz(sz==0) = 1;
        Y = nan(sz);
        X.(fns{i}) = Y;
      end                        
    end  
  end        
end












