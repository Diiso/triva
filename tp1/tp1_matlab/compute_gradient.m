function [dIx dIy dI_norm dI_orientation]=compute_gradient(I)
% Compute the gradient of I by finite differences and return
%   - dIx the gradient along the first dimension
%   - dIy the gradient along the second dimension
%   - dI_norm the norm of the gradient
%   - dI_orientation the orientation of the gradient
% Note:since the gradient is computed by finite differences, it 
% cannot work at the border of the image. Have all the outputs be of size
% (size(I)-1)
% Warning: the first dimension in Matlab is vertical (y)
% Tip: for the orientation, use 'atan' function

dIx = zeros(size(I)-1);
dIy = zeros(size(I)-1);
dI_norm = zeros(size(I)-1);
dI_orientation = zeros(size(I)-1);

[H,W] = size(I)
for i=2:(H-1)
  for j=2:(W-1)
    dIx(i,j) = I(i+1,j)-I(i-1,j);
    dIy(i,j) = I(i,j+1)-I(i,j-1);
  end
end

for i=2:H-1
  for j=2:W-1
    dI_norm(i,j) = sqrt(dIx(i,j).^2+dIy(i,j).^2);
    dI_orientation(i,j) = atan(dIy(i,j)/(0.0000001+dIx(i,j)));
  end
end
end