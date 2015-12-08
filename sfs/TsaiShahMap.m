function [ map ] = TsaiShahMap( image )
% Creates a DEM from a set of asteroid/comet images using shape from
% shading proposed by Tsai and Shah (Shape From Shading Using Linear Approximation).
% Link to paper: http://crcv.ucf.edu/papers/ivcj94.pdf
%
% Input
%   image : An image to create DEM from.
% Output
%   map    : The DEM from given image

I = double(mat2gray(image));
 
[num_rows,num_cols] = size(I);

Z_n = zeros(num_rows,num_cols); % up-to-date depth values
Z_n_1 = zeros(num_rows,num_cols); % depth values of previous iteration
S = zeros(num_rows,num_cols) + 0.01;

[slant,tilt] = SlantTiltEstimation(I);

i_x = cos(tilt) * tan(slant);
i_y = sin(tilt) * tan(slant);

num_iterations = 200;
 
for iterations = 1:num_iterations
    for i = 1:num_rows
        for j = 1:num_cols
            p = 0.0;
            q = 0.0;
            
            if j - 1 >= 1 && i - 1 >= 1
                p = Z_n_1(i,j) - Z_n_1(i, j-1);
                q = Z_n_1(i,j) - Z_n_1(i-1, j);
            end
            
            f = -(I(i,j) - max(0.0,(1+p*i_x+q*i_y)/(sqrt(1+p^2+q^2)*sqrt(1+i_x^2+i_y^2))));
            df = -((i_x+i_y)/(sqrt(1+p^2+q^2)*sqrt(1+i_x^2+i_y^2))-(p+q)*(1.0+p*i_x+q*i_y)/(sqrt((1+p^2+q^2)^3)*sqrt(1+i_x^2+i_y^2)));
            
            K = S(i,j)*df/(eps+S(i,j)*df^2);
            
            S(i,j) = S(i,j) - S(i,j)*K*df;
            
            Z_n(i,j) = Z_n_1(i,j) + K*f;
        end
    end
    
    Z_n_1 = Z_n;
end

map = medfilt2(abs(Z_n),[21 21]);