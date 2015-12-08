function [slant,tilt] = SlantTiltEstimation(img)

% Compute gradient
[Ex,Ey] = gradient(img);

% normalize the gradients to be unit vectors
Exy = sqrt(Ex.^2 + Ey.^2);
Ex = Ex ./ (Exy + eps);
Ey = Ey ./ (Exy + eps);

% computing the average of the normalized gradients
avgEx = mean(Ex(:));
avgEy = mean(Ey(:));

% estimate tilt
tilt = atan(avgEy/avgEx);
if tilt < 0
    tilt = tilt + pi;
end

% first moment of image intensities
Mu1 = mean(img(:));
% second moment of image intensities
Mu2 = mean(mean(img.^2));

% estimate slant
slant = 0.0;
min_diff = Inf;
for x=0:0.0001:(pi/2)
    result = 0.5577 + 0.6240 * cos(x) + 0.1882 * (cos(x))^2 - 0.6514 * (cos(x))^3 - 0.5345 * (cos(x))^4 + 0.9282 * (cos(x))^5 + 0.3476 * (cos(x))^6 - 0.4984 * (cos(x))^7;
    diff = abs(Mu1 / sqrt(Mu2) - result);
    if diff < min_diff
        min_diff = diff;
        slant = x;
    end
end

end