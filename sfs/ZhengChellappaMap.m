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

image = image ./ max(image(:));

% set parameters
num_of_iterations = 10;
p_q_delta = 0.0001;
mu = 1;
albedo = 0.05;

% estimate light direction
[slant, tilt] = SlantTiltEstimation(image);
light_direction = [cos(tilt)*sin(slant) sin(tilt)*sin(slant) cos(slant)];

% determine number of hierarchies
num_of_hierarchical_levels = 1;
[img_height,img_width] = size(image);
min_dim = min(img_height, img_width);
while min_dim > 64
    min_dim = min_dim / 2 + rem(min_dim, 2);
    num_of_hierarchical_levels = num_of_hierarchical_levels + 1;
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
    
    [p_x,p_y] = gradient(p{h});
    [p_xx,~] = gradient(p_x);
    [~,p_yy] = gradient(p_y);
    
    [q_x,q_y] = gradient(q{h});
    [q_xx,~] = gradient(q_x);
    [~,q_yy] = gradient(q_y);
    
    [Z_x,Z_y] = gradient(Z{h});
    [Z_xx,~] = gradient(Z_x);
    [~,Z_yy] = gradient(Z_y);
    
    [current_height, current_width] = size(images{h});
    
    for t=1:num_of_iterations
        R = albedo .* (-light_direction(1).* p{h} - light_direction(2).* q{h} + light_direction(3))./sqrt(1 + p{h}.^2 + q{h}.^2);
        R = max(0,R);
        
        R_p = albedo .* (-light_direction(1).* (p{h} + p_q_delta) - light_direction(2).* q{h} + light_direction(3))./sqrt(1 + (p{h} + p_q_delta).^2 + q{h}.^2);
        R_p = max(0,R_p);
        R_p = (R_p - R) ./ p_q_delta;
        
        R_q = albedo .* (-light_direction(1).* p{h} - light_direction(2).* (q{h} + p_q_delta) + light_direction(3))./sqrt(1 + p{h}.^2 + (q{h} + p_q_delta).^2);
        R_q = max(0,R_q);
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
        
        for i=1:current_height
            for j=1:current_width
                if isfinite(d_Z(i,j))
                    p{h}(i,j) = p{h}(i,j) + d_p(i,j);
                    q{h}(i,j) = q{h}(i,j) + d_q(i,j);
                    Z{h}(i,j) = Z{h}(i,j) + d_Z(i,j);
                end
            end
        end
    end
    
    if h == 1
        break
    end
    
    [height, width] = size(images{h-1});
    
    for i=1:height
        for j=1:width
            i_half = round(i/2);
            j_half = round(j/2);
            
            if rem(i,2) == 1 && rem(j,2) == 1
                p{h-1}(i,j) = p{h}(i_half,j_half);
                q{h-1}(i,j) = q{h}(i_half,j_half);
                Z{h-1}(i,j) = Z{h}(i_half,j_half);
                
                if i_half <= current_height && j_half + 1 <= current_width
                    p{h-1}(i,j) = p{h-1}(i,j) + p{h}(i_half,j_half + 1);
                    q{h-1}(i,j) = q{h-1}(i,j) + q{h}(i_half,j_half + 1);
                    Z{h-1}(i,j) = Z{h-1}(i,j) + Z{h}(i_half,j_half + 1);
                end
                if i_half + 1 <= current_height && j_half <= current_width
                    p{h-1}(i,j) = p{h-1}(i,j) + p{h}(i_half + 1,j_half);
                    q{h-1}(i,j) = q{h-1}(i,j) + q{h}(i_half + 1,j_half);
                    Z{h-1}(i,j) = Z{h-1}(i,j) + Z{h}(i_half + 1,j_half);
                end
                if i_half + 1 <= current_height && j_half + 1 <= current_width
                    p{h-1}(i,j) = p{h-1}(i,j) + p{h}(i_half + 1,j_half + 1);
                    q{h-1}(i,j) = q{h-1}(i,j) + q{h}(i_half + 1,j_half + 1);
                    Z{h-1}(i,j) = Z{h-1}(i,j) + Z{h}(i_half + 1,j_half + 1);
                end
                
                p{h-1}(i,j) = p{h-1}(i,j) * 0.25;
                q{h-1}(i,j) = q{h-1}(i,j) * 0.25;
                Z{h-1}(i,j) = Z{h-1}(i,j) * 0.5;
            else
                if rem(j,2) == 1
                    p{h-1}(i,j) = p{h}(i_half,j_half);
                    q{h-1}(i,j) = q{h}(i_half,j_half);
                    Z{h-1}(i,j) = Z{h}(i_half,j_half);
                    
                    if i_half <= current_height && j_half + 1 <= current_width
                        p{h-1}(i,j) = p{h-1}(i,j) + p{h}(i_half,j_half + 1);
                        q{h-1}(i,j) = q{h-1}(i,j) + q{h}(i_half,j_half + 1);
                        Z{h-1}(i,j) = Z{h-1}(i,j) + Z{h}(i_half,j_half + 1);
                    end
                    
                    p{h-1}(i,j) = p{h-1}(i,j) * 0.5;
                    q{h-1}(i,j) = q{h-1}(i,j) * 0.5;
                else
                    if rem(i,2) == 1
                        p{h-1}(i,j) = p{h}(i_half,j_half);
                        q{h-1}(i,j) = q{h}(i_half,j_half);
                        Z{h-1}(i,j) = Z{h}(i_half,j_half);
                        
                        if i_half + 1 <= current_height && j_half <= current_width
                            p{h-1}(i,j) = p{h-1}(i,j) + p{h}(i_half + 1,j_half);
                            q{h-1}(i,j) = q{h-1}(i,j) + q{h}(i_half + 1,j_half);
                            Z{h-1}(i,j) = Z{h-1}(i,j) + Z{h}(i_half + 1,j_half);
                        end
                        
                        p{h-1}(i,j) = p{h-1}(i,j) * 0.5;
                        q{h-1}(i,j) = q{h-1}(i,j) * 0.5;
                    else
                        p{h-1}(i,j) = p{h}(i_half,j_half);
                        q{h-1}(i,j) = q{h}(i_half,j_half);
                        Z{h-1}(i,j) = Z{h}(i_half,j_half) * 2;
                    end
                end
            end
        end
    end
end

map = Z{1};

