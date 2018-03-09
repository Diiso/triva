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

end