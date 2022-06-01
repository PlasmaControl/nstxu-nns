function s = struct_fields_to_double(s)
  fnames = fieldnames(s);
  for i = 1:length(fnames)
    fn = fnames{i};
    if isa(s.(fn), 'single')
      s.(fn) = double(s.(fn));
    end
  end
end
