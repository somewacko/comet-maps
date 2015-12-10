function [ map ] = ZhengChellappaMap( image )
% Creates a DEM from a set of asteroid/comet images using shape from
% shading proposed by Zheng and Chellappa.
%
% Input
%   image : An image to create DEM from.
% Output
%   map    : The DEM from given image

if size(image, 3) == 3
    image = double(rgb2gray(image));
else
    image = double(mat2gray(image));
end

% set parameters
num_of_iterations = 500;
p_q_delta = 0.0001;
mu = 1;
albedo = 0.01;

% normalize image
image = image ./ max(image(:));
image = image ./ albedo;

% estimate light direction
[slant, tilt] = SlantTiltEstimation(image);
light_direction = [cos(tilt)*sin(slant) sin(tilt)*sin(slant) -cos(slant)];

norm_factor = sqrt(light_direction(1)^2 + light_direction(2)^2 + light_direction(3)^2);
light_direction = [light_direction(1) / norm_factor light_direction(2) / norm_factor light_direction(3) / norm_factor];

% determine number of hierarchies
num_of_hierarchical_levels = 1;
[img_height,img_width] = size(image);
min_dim = min(img_height, img_width);
while min_dim > 64
    min_dim = min_dim / 2 + rem(min_dim, 2);
    %num_of_hierarchical_levels = num_of_hierarchical_levels + 1;
end

% create hierarchies
for h=1:num_of_hierarchical_levels
    images{h} = imresize(image, 0.5^(h-1));
    
    [height, width] = size(images{h});
    
    p{h} = zeros(height,width);
    q{h} = zeros(height,width);
    Z{h} = zeros(height,width);
end

% start iteration for each hierarchical level
for h=num_of_hierarchical_levels:-1:1
    [I_x,I_y] = imgradientxy(images{h});
    [I_xx,~] = imgradientxy(I_x);
    [~,I_yy] = imgradientxy(I_y);
    
    [p_x,p_y] = imgradientxy(p{h});
    [p_xx,~] = imgradientxy(p_x);
    [~,p_yy] = imgradientxy(p_y);
    
    [q_x,q_y] = imgradientxy(q{h});
    [q_xx,~] = imgradientxy(q_x);
    [~,q_yy] = imgradientxy(q_y);
    
    [Z_x,Z_y] = imgradientxy(Z{h});
    [Z_xx,~] = imgradientxy(Z_x);
    [~,Z_yy] = imgradientxy(Z_y);
    
    for t=1:num_of_iterations
        R = (-light_direction(1).* p{h} - light_direction(2).* q{h} + light_direction(3))./ sqrt(1 + p{h}.^2 + q{h}.^2);
        
        R_p = (-light_direction(1).* (p{h} + p_q_delta) - light_direction(2).* q{h} + light_direction(3))./ sqrt(1 + (p{h} + p_q_delta).^2 + q{h}.^2);
        R_p = (R_p - R) ./ p_q_delta;
        
        R_q = (-light_direction(1).* p{h} - light_direction(2).* (q{h} + p_q_delta) + light_direction(3))./ sqrt(1 + p{h}.^2 + (q{h} + p_q_delta).^2);
        R_q  = (R_q - R) ./ p_q_delta;
        
        A_11 = 5 .* R_p.^2 + 1.25 .* mu;
        A_12 = 5 .* R_p .* R_q + 0.25 .* mu;
        A_22 = 5 .* R_q.^2 + 1.25 .* mu;
        
        lunate_eps = R - images{h};
        lower_eps = R_p .* (p_xx + p_yy) + R_q .* (q_xx + q_yy) - I_xx - I_yy;
        
        C_3 = -p_x - q_y + Z_xx + Z_yy;
        C_1 = (lower_eps - lunate_eps) .* R_p - mu .* (p{h} - Z_x) - 0.25 .* mu .* C_3;
        C_2 = (lower_eps - lunate_eps) .* R_q - mu .* (q{h} - Z_y) - 0.25 .* mu .* C_3;
        
        determinant = A_11 .* A_22 - A_12.^2;
        
        d_p = (C_1 .* A_22 - C_2 .* A_12) ./ determinant;
        d_q = (C_2 .* A_11 - C_1 .* A_12) ./ determinant;
        d_Z = (C_3 + d_p + d_q) ./ 4;
        
        p{h} = p{h} + d_p;
        q{h} = q{h} + d_q;
        Z{h} = Z{h} + d_Z;
    end
    
    if h == 1
        break
    end
    
    p{h-1} = imresize(p{h}, size(p{h-1}));
    q{h-1} = imresize(q{h}, size(q{h-1}));
    Z{h-1} = imresize(Z{h}, size(Z{h-1}));
end

map = Z{1};

