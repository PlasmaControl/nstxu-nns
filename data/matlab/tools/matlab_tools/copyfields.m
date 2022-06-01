% Copies the fields of B to the fields of A. If overwrite=1, the fields of 
% B will overwrite contents of corresponding field in A. If overwrite=0, 
% only fields of B that do not exist in A will be copied. 
%
% Use fields2copy = [] to copy all fields, or let fields2copy be a cell
% array of strings. 
function A = copyfields(A,B,fields2copy,overwrite)

if isempty(fields2copy), fields2copy = fieldnames(B); end
if ~exist('overwrite', 'var'), overwrite = 0; end

for i = 1:length(fields2copy)
  fn = fields2copy{i};  
  if ~isfield(A,fn) || overwrite
    A.(fn) = B.(fn);
  end
end
