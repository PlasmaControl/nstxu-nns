
for i = 1:circ.nvx
  
  k = find(circ.vvgroup == i, 1);
  z = tok_data_struct.vvdata(1,k);
  r = tok_data_struct.vvdata(2,k);
  text(r,z,[num2str(i) ' ' circ.vvnames{i}])
end
