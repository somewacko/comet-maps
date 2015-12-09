function [ map ] = LeeRosenfeldMap( image )
% Creates a DEM from a set of asteroid/comet images using shape from
% shading proposed by Lee and Rosenfeld (Improved Methods of Estimating Shape
% from Shading Using the Light Source Coordinate System).
%
% Input
%   image : An image to create DEM from.
% Output
%   map    : The DEM from given image

if size(image, 3) == 3
    I = double(rgb2gray(image));
else
    I = double(mat2gray(image));
end

I = I ./ max(I(:));

[num_rows,num_cols] = size(I);

Z = zeros(num_rows, num_cols);

[slant_L,tilt_L] = SlantTiltEstimation(I);

% coordinate transformation matrix from viwer coordinate system to
% illumination coordinate system
coord_transformation = [cos(slant_L)*cos(tilt_L) cos(slant_L)*sin(tilt_L) -sin(slant_L); -sin(tilt_L) cos(tilt_L) 0; sin(slant_L)*cos(tilt_L) sin(slant_L)*sin(tilt_L) cos(slant_L)];

for i = 1:num_rows-1
    for j = 1:num_cols-1
        Ix = -(I(i,j+1) - I(i,j));
        Iy = I(i+1,j) - I(i,j);
        Iz = 0;
        
        II = coord_transformation * [Ix; Iy; Iz];
        
        % slant and tilt of the surface normal viewed from the illumination coordinate system
        slant_S = acos(I(i,j)); % from section 3
        tilt_S = atan2(II(2),II(1)+eps); % from theorem 2.1
        
        while tilt_S < 0
            tilt_S = tilt_S + 2*pi; % make tilt positive
        end
        
        % z' axis in illumination coordinate system (i.e. illumination direction)
        z_prime_axis = [sin(slant_S)*cos(tilt_S); sin(slant_S)*sin(tilt_S); cos(slant_S)];
        
        % z axis in viewer's coordinate system
        z_axis = coord_transformation \ z_prime_axis;
        
        Z(i,j) = z_axis(3) / (sqrt(z_axis(1)^2 + z_axis(2)^2 + z_axis(3)^2) + eps);
    end
end

maxZ = max(Z(:));
minZ = min(Z(:));

map = (Z - minZ) .* 10 ./ (maxZ-minZ);