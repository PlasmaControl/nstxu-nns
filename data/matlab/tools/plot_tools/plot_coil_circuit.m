% icoils = index of coils to plot [ecoils, fcoils, vacuum vessel circuits]

function plot_coil_circuit(icoils, tok_data_struct, rgb)

if nargin < 3, rgb = [0 0 0]; end

struct_to_ws(tok_data_struct);
nec = size(ecnames,1);


for icoil = icoils
  
  % Plot for an E-coil
  if icoil <= nec
    
    k = find( ecdata(5,:) == icoil);
    
    zbotleft  = min(ecdata(1,k)) - ecdata(3,1)/2;
    rbotleft  = min(ecdata(2,k)) - ecdata(4,1)/2;
    ztopleft  = max(ecdata(1,k)) + ecdata(3,1)/2;
    rbotright = max(ecdata(2,k)) + ecdata(4,1)/2;
    
    dz = ztopleft  - zbotleft;
    dr = rbotright - rbotleft;
    
    position = [rbotleft zbotleft dr dz];
        
    rectangle('Position', position, 'FaceColor', rgb, 'EdgeColor', rgb)
    rmid = rbotleft + dr/2;
    zmid = zbotleft + dz/2;
    
  % Plot for an F-Coil
  elseif icoil <= nc 
  
    ii = icoil - nec;
    
    zbotleft = fcdata(1,ii) - fcdata(3,ii)/2;
    rbotleft = fcdata(2,ii) - fcdata(4,ii)/2;
    position = [rbotleft zbotleft fcdata(4,ii) fcdata(3,ii)];
    rectangle('Position', position, 'FaceColor', rgb, 'EdgeColor', rgb)      
    
  % Plot for a vacuum vessel element
  else
    ii = icoil - nc;
    
    z   = vvdata(1,ii);
    r   = vvdata(2,ii);
    dz  = vvdata(3,ii);
    dr  = vvdata(4,ii);
    ac  = vvdata(5,ii);
    ac2 = vvdata(6,ii);
    
    plot_efit_region(z, r, dz, dr, ac, ac2, rgb);
    
    scatter(r,z,100,'filled','markerfacecolor', rgb)
    
  end
end
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
