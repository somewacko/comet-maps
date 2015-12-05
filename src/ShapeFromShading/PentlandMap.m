function [ map ] = PentlandMap( image )
% Creates a DEM from a set of asteroid/comet images using shape from
% shading proposed by Tsai and Shah.
% Code from http://blog.tibarazmi.com/shape-shading-pentland-approach/
%
% Input
%   image : An image to create DEM from.
% Output
%   map    : The DEM from given image

% making sure that it is a grayscale image
E = mat2gray(image);
E = double(E);
% normalizing the image to have maximum of one
E = E ./ max(E(:));
%E = boost_shadow(E); imshow(E);
% first compute the surface albedo and illumination direction
[slant,tilt] = SlantTiltEstimation (E);
 
% compute the fourier transform of the image
Fe = fft2(E);
 
% wx and wy
[M,N] = size(E);
[x,y] = meshgrid(1:N,1:M);
wx = (2.* pi .* x) ./ M;
wy = (2.* pi .* y) ./ N;
 
% Using the estimated illumination direction
Fz =Fe./(-1i.*wx.*cos(tilt).*sin(slant)-1i.*wy.*sin(tilt).*sin(slant));
 
% Compute the inverse Fourier transform to recover the surface.
map = abs(ifft2(Fz));