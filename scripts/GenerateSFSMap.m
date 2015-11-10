% GenerateSFSMap.m
%
% Driver script to generate DEMs from asteroid/comet images using the
% shape from shading function.

clc; clf;

% Read comet image
image = imread('20150427101235-7719c656-xl.png');

% Create depth map from SFS algorithm
map = ZhengChellappaMap(image);

% Performing smoothing of recovered surface
depth_map = medfilt2(abs(map),[21 21]);

% Draw 3D surface
figure;
h = surfl(depth_map);
set(h,'ydata',flipud(get(h,'ydata')));
set(h,'xdata',flipud(get(h,'xdata')));
shading interp;
colormap gray(256);
lighting phong;

