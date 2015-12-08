function [ map ] = TsaiShahMap( image )
% Creates a DEM from a set of asteroid/comet images using shape from
% shading proposed by Tsai and Shah.
% Code from http://blog.tibarazmi.com/shape-shading-tsai-shah-approach/
%
% Input
%   image : An image to create DEM from.
% Output
%   map    : The DEM from given image

% making sure that it is a grayscale image
E = mat2gray(image);
E = double(E);
% first compute the surface albedo and illumination direction
[slant,tilt] = SlantTiltEstimation(E);
 
% initializations
[M,N] = size(E);
% surface normals
p = zeros(M,N);
q = zeros(M,N);
% the surface
Z = zeros(M,N);
% surface derivatives in x and y directions
Z_x = zeros(M,N);
Z_y = zeros(M,N);
% maximum number of iterations
maxIter = 200;
% the normalized illumination direction
ix = cos(tilt) * tan(slant);
iy = sin(tilt) * tan(slant);
 
for k = 1 : maxIter
    % using the illumination direction and the currently estimate
    % surface normals, compute the corresponding reflectance map.
    R =(cos(slant) + p .* cos(tilt)*sin(slant)+ q .* ...
        sin(tilt)*sin(slant))./sqrt(1 + p.^2 + q.^2);
    % at each iteration, make sure that the reflectance map is positive at
    % each pixel, set negative values to zero.
    R = max(0,R);
    % compute our function f which is the deviation of the computed
    % reflectance map from the original image ...
    f = E - R;
    % compute the derivative of f with respect to our surface Z
    df_dZ =(p+q).*(ix*p + iy*q + 1)./(sqrt((1 + p.^2 + q.^2).^3)* ...
        sqrt(1 + ix^2 + iy^2))-(ix+iy)./(sqrt(1 + p.^2 + q.^2)* ...
        sqrt(1 + ix^2 + iy^2));
    % update our surface
    Z = Z - f./(df_dZ + eps); % to avoid dividing by zero
    % compute the surface derivatives with respect to x and y
    Z_x(2:M,:) = Z(1:M-1,:);
    Z_y(:,2:N) = Z(:,1:N-1);
    % using the updated surface, compute new surface normals
    p = Z - Z_x;
    q = Z - Z_y;
end
 
% smoothing the recovered surface
map = medfilt2(abs(Z),[21 21]);