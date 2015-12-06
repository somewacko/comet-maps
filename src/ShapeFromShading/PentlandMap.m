function [ map ] = PentlandMap( image )
% Creates a DEM from a set of asteroid/comet images using shape from
% shading proposed by Pentland (1988).
%
% Input
%   image : An image to create DEM from.
% Output
%   map    : The DEM from given image

I = double(mat2gray(image));
I = I ./ max(I(:));

[num_rows,num_cols] = size(I);

[slant,tilt] = SlantTiltEstimation(I);
 
% Compute Fourier transform of the image
F_I = fft2(I);

% Compute Fourier transform of the depth map given FT of image
F_Z = zeros(num_rows, num_cols);

for i=1:num_rows
    for j=1:num_cols
        F_Z(i,j) = F_I(i,j) / (-1i * ((2 * pi * j / num_rows) * cos(tilt).*sin(slant) + (2 * pi * i / num_rows) * sin(tilt) * sin(slant)));
    end
end

% Compute inverse Fourier transform to recover depth map
map = abs(ifft2(F_Z));