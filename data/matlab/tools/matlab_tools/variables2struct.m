function s = variables2struct(varargin)
  s = struct;
  for i = 1:nargin
    s.(inputname(i)) = varargin{i};
  end
end












