% unpack the vector/struct/cell v into multiple output arguments

% example: v = [1 2 3]
%          [a,b,c,] = unpack(v)
%
%          returns a=1,b=2,c=3

function [varargout] = unpack(v)
    if isnumeric(v), v=num2cell(v); end
    if isstruct(v), v=struct2cell(v); end
        
    for i = 1:numel(v)
        varargout{i} = v{i};
    end
end