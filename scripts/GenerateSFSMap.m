% GenerateSFSMap.m
%
% Driver script to generate DEMs from asteroid/comet images using the
% shape from shading function.

clc; clf;

% Read comet image
image = imread('mozart.png');

% Create depth map from SFS algorithm
depth_map = TsaiShahMap(image);

max_z = max(max(depth_map)) * 1.5;
min_z = min(min(depth_map));

% Draw 3D surface
h = surfl(depth_map); % use 'mesh' function (instead of 'surfl') for comet images 
zlim([min_z max_z]);
set(h,'ydata',flipud(get(h,'ydata')));
set(h,'xdata',flipud(get(h,'xdata')));
shading interp;
colormap gray(256);
%lighting phong;
view([-40 75]); % [0 83] for comet images

