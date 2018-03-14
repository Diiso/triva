%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TP1: Canny edge detection and Bilateral filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Beginning with Matlab
% Read and show an image
Irgb = imread('images/tools.jpg'); % read the image
% Images are read as matrix, Irgb(2,1,1) means the red channel of Irgb at x=1,y=2
[h,w,c]=size(Irgb);
fprintf('The size of Irgb is %i x %i x %i of type %s \n',h,w,c,class(Irgb));
fprintf('Its values are between %i and %i \n',min(Irgb(:)), max(Irgb(:)));
Irgb=double(Irgb)./255;
fprintf('Irgb is now of type %s \n',class(Irgb));
fprintf('Its values are between %i and %i \n\n',min(Irgb(:)), max(Irgb(:)));

I=get_luminance(Irgb); % convert the color image into grayscale
[h,w,c]=size(I);
fprintf('The size of I is %i x %i x %i of type %s \n',h,w,c,class(I));
fprintf('Its values are between %i and %i \n\n',min(I(:)), max(I(:)));

figure(1); % open a new figure with the handle 1
subplot(1,2,1); % divide the figure into a 1x2 array and use the first cell
imagesc(Irgb);
axis image off;
title('RGB image'); % add a title to the first cell

subplot(1,2,2); % use the second cell
imagesc(I);
colormap(gray);
axis image off;
title('B&W image');

if ~exist('results')
    mkdir('results')
end

imwrite(I,'results/black_and_white_image.png'); % save the black and white image
%print(1,'results/figure_1.jpg','-djpeg'); % save figure 1
% close(1); in case you want matlab to close figure 1

% add gaussian noise to the image
sigma_noise=0.1;
I_noise=I+randn(size(I))*sigma_noise; % This is the blurred image

figure(2);

subplot(1,2,1);
imagesc(I);
axis image off;
colormap(gray);
title('B&W Image');

subplot(1,2,2);
imagesc(I_noise);
axis image off;
colormap(gray);
title('Image with Noise');

imwrite(I_noise,'results/image_with_noise.png'); % save the black and white image


%% 2. Basic Image Processing
% Gaussian convolution
I_blurred=gaussian_convolution(I,2); 
figure(3);
imagesc(I_blurred);
axis image off;
colormap(gray);
title('Blurred Image')



% Gradients
[dIx dIy dI_norm dI_orientation]=compute_gradient(I_blurred);


% Visualize thresholded gradients
threshold=0.01;
dI_norm_thresh=dI_norm>threshold;
figure(4);
imagesc(dI_norm_thresh);
axis image off;
colormap(gray);
title(sprintf('Gradients bigger than %.03f',threshold));


%% 3. Canny edge detector
% Quantify gradient orientations
quantified_orientation=quantify_gradient(dI_orientation);

% This is just to check that your results are meaningful, use it!
orientation=3;
figure(5);
imagesc(quantified_orientation==orientation);
axis image off;
colormap(gray);
title(sprintf('Orientations quantified to bin %i',orientation));


% Perform non-max suppression
nms_edges=non_max_suppression(dI_norm,quantified_orientation,threshold);
figure(6)
imagesc(nms_edges);
axis image off;
colormap(gray);
title(sprintf('Non-max Supression'));

% Canny edges
% parameters
sigma = 2; % Deviation standard du flou gaussien
s1 = 0.01; % Seuil haut de l'hysteresis
s2 =  0.002; % Seuil bas  de l'hysteresis

edges=canny_edges(I,sigma,s1,s2);
figure(7)
imagesc(edges);
axis image off;
colormap(gray);
title(sprintf('Canny Edges Detection'));

%% 4. Bilateral Filter

% Edge-aware smoothing
Irgb = double(imread('images/rock2.png'))./255; % read the image
I=get_luminance(Irgb); % get luminance
rgb_ratio=Irgb./repmat(I,[1 1 3]); % compute image color
smoothed_image=BF(I,5,0.1);
output=max(0,min(1,repmat(smoothed_image, [1 1 3]).*rgb_ratio));%
figure(1)
imagesc(output,[0 1]);
axis image off;
title('Smoothed image');

% Detail enhancement


%% 5. Optionnal
