function s = append2struct(s, varargin)  
  for i = 2:nargin
    s.(inputname(i)) = varargin{i-1};
  end
end












