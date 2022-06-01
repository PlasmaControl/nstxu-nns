clear; clc; close all

% ffprim = load('/p/nstxusr/nstx-users/jwai/nstxu-nns/data/rawdata/data_by_var/train/ffprim.mat');
% y = ffprim.ffprim;
% ncoeffs = 3;

pprime = load('/p/nstxusr/nstx-users/jwai/nstxu-nns/data/rawdata/data_by_var/train/pprime.mat');
y = pprime.pprime;
ncoeffs = 2;

x = linspace(0,1,size(y,2));

for i = 1:size(y,1)  
  p(i,:) = polyfit(x, y(i,:), ncoeffs);
  
end



% i = 100;
% y1 = polyval(p(i,:), x);
% 
% figure
% hold on
% plot(x, y(i,:))
% scatter(x, y1)



