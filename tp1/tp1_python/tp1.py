
# coding: utf-8

# # TP1 TRIVA - Louis Montaut

# This project is made using Python. I did the same thing with Octave (Matlab), but I spent less time on it since I wanted to exercise my python skills.<br>
# You can check out my matlab version, but this python version is the most advanced of the two.

# ##  Important presets

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# In[2]:


file_name = 'images/tools.jpg'

#We need to get the format of the file because python treats png and jpg in a different way !
file_format = file_name[len(file_name)-3]+file_name[len(file_name)-2]+file_name[len(file_name)-1]


# In[3]:


def showImages(ImagesArray,name_of_file = '',saveFile = 'false'):
    fig = plt.figure(figsize=(13,13))
    
    n = len(ImagesArray)
    rows = n//2+1
    for i in range(n):
            fig.add_subplot(rows,2,i+1)
            plt.axis('off')
            plt.title(ImagesArray[i][1])
            plt.imshow(ImagesArray[i][0], cmap = plt.get_cmap('gray'))
    plt.show()
    
    if saveFile:
        #Do not forget to add the extension in the name_of_file
        fig.savefig('results/'+name_of_file+'.'+file_format)
    return None


# In[4]:


if not os.path.exists("results"):
    os.makedirs("results")


# In[5]:


def get_luminance(Irgb):
    return np.dot(Irgb[...,:3], [0.2126, 0.7152, 0.0722])


# ## I) Basic image processing

# ### RGB Image

# In[6]:


Irgb = mpimg.imread(file_name)
fig = plt.figure(figsize=(6,6))
plt.axis('off')
plt.title('RGB image')
imgplot = plt.imshow(Irgb)
plt.show()


# ### Black and white

# In[7]:


I = get_luminance(Irgb)
showImages([(I,'B&W image')],'black_and_white',True)


# ### Noise

# **Why is there a different sigma_noise depending on jpg or png?**<br>
# Because Python doesn't interpret PNG and JPG files the same way. To python, a PNG file has grayscale values between 0 and 1. JPG files have grayscale values between 0 and 255.<br>
# **When we are looking at a JPG file, it is normal for the sigma_noise to be above 3. Divide this value by 255 if you want to get the same value for a PNG file.**
# **A PNG file will have a "strong" with sigma_noise>0.05**

# In[8]:


sigma_noise_1 = 0.05
sigma_noise_2 = 0.1
sigma_noise_3 = 0.3

if file_format == 'jpg':
    sigma_noise_1*=255
    sigma_noise_2*=255
    sigma_noise_3*=255

n,m = I.shape
I_noise_1 = I + np.random.randn(n,m)*sigma_noise_1
I_noise_2 = I + np.random.randn(n,m)*sigma_noise_2
I_noise_3 = I + np.random.randn(n,m)*sigma_noise_3

showImages([(I,'B&W image'),(I_noise_1,'Image with noise = '+str(sigma_noise_1)),
            (I_noise_2,'Image with noise = '+str(sigma_noise_2)),
            (I_noise_3,'Image with noise = '+str(sigma_noise_3))],'noise',True)


# ### Gaussian convolution

# For the Gaussian convolution, it is important to check if the integral (sum in this case) of the kernel we use to calculate the blurr is equal to 1. Here, we wrote the code in **gaussian_filter_2d.py** to match this property.

# In[9]:


import gaussian_convolution as gc
sigma_1 = 2
sigma_2 = 6
I_blurred_1 = gc.gaussianConvolution(I,sigma_1)
I_blurred_2 = gc.gaussianConvolution(I,sigma_2)

showImages([(I,'B&W image'),(I_blurred_1,'Blurred image with sigma = '+str(sigma_1)),
            (I_blurred_2,'Blurred image with sigma = '+str(sigma_2))],'gaussian_convolution',True)


# In Python, we can choose how the convolution is calculated on the edges. Here, I chose the parameter "symm" for the boundaries (check in **gaussian_convolution.py**). Therefore, the algorithm "prolongates" the boundaries of the image in order to calculate the blur.<br>
# If we don't do so, we get dark borders.

# ### Gradient

# **Why is there a different threshold depending on jpg or png?**<br>
# Because Python doesn't interpret PNG and JPG files the same way. To python, a PNG file has grayscale values between 0 and 1. JPG files have grayscale values between 0 and 255.<br>
# **When we are looking at a JPG file, it is normal for the threshold to be above 10. Divide this value by 255 if you want to get the same value for a PNG file.**

# In[10]:


threshold = 0.05

if file_format == 'jpg':
    threshold*=255


# Here, I made the code so that we can compute the gradient using two techniques.<br>
# The first one : **computeGradientNoConvolution** doesn't use kernels to compute the gradient. It just uses the standard finite differences.<br>
# The second one : **computeGradient** uses convolution to compute the gradient.

# In[11]:


from gradient import computeGradient, computeGradientNoConvolution

I_blurred = gc.gaussianConvolution(I,2)
I_gradient = computeGradientNoConvolution(I_blurred)
#I_gradient = computeGradient(I_blurred)
dIx,dIy,dI_norm,dI_orientation = I_gradient.dIx,I_gradient.dIy,I_gradient.dI_norm,I_gradient.dI_orientation

n,m = dI_norm.shape
dI_norm_threshold = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        if dI_norm[i,j]>threshold:
            dI_norm_threshold[i,j]=dI_norm[i,j]

threshold_2 = 0.1
threshold_3 = 0.2

if file_format == 'jpg':
    threshold_2 = 0.1*255
    threshold_3 = 0.2*255

dI_norm_threshold_2 = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        if dI_norm[i,j]>threshold_2:
            dI_norm_threshold_2[i,j]=dI_norm[i,j]


dI_norm_threshold_3 = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        if dI_norm[i,j]>threshold_3:
            dI_norm_threshold_3[i,j]=dI_norm[i,j]
            
showImages([(dI_norm,'Gradient norm'),
            (dI_norm_threshold,'Gradient norm with threshold = '+str(threshold)),
           (dI_norm_threshold_2,'Gradient norm with threshold = '+str(threshold_2)),
           (dI_norm_threshold_3,'Gradient norm with threshold = '+str(threshold_3))],
           'gradient_norm',True)


# ## II) Canny Edge Detector

# ### Quantify Gradient

# In[12]:


from quantify_gradient import quantifyGradient
I_quantified_gradient = quantifyGradient(dI_orientation)

orientation = 2

n,m = I_quantified_gradient.shape
I_orientation = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        if I_quantified_gradient[i,j] == orientation:
            I_orientation[i,j] = 1
        else :
            I_orientation[i,j] = 0
            

showImages([(dI_norm,'Gradient norm'),(I_orientation,'Orientation = '+str(orientation))],'orientation',True)
            


# ### Non-max Suppression

# In[13]:


from non_max_supression import nms
I_nms_1 = nms(dI_norm, I_quantified_gradient, threshold)
I_nms_2 = nms(dI_norm, I_quantified_gradient, threshold*2)

showImages([(I_nms_1,'Non-max suppression with threshold = ' + str(threshold)),
            (I_nms_2,'Non-max suppresion with 2*threshold = '+str(2*threshold))],'Non_max_suppresion',True)


# With the non-max suppression, we already start to get some good idea of the edges. Now each of them are only a pixel wide.<br>
# **Here is the effect of noise on the non-max suppression : **

# In[14]:


I_blurred_noise_1 = gc.gaussianConvolution(I_noise_2,2)
I_blurred_noise_2 = gc.gaussianConvolution(I_noise_3,2)

I_gradient_noise_1 = computeGradientNoConvolution(I_blurred_noise_1)
I_gradient_noise_2 = computeGradientNoConvolution(I_blurred_noise_2)

dIn1_norm = I_gradient_noise_1.dI_norm
dIn1_orientation = I_gradient_noise_1.dI_orientation
dIn2_norm = I_gradient_noise_2.dI_norm
dIn2_orientation = I_gradient_noise_2.dI_orientation

I_quantified_gradient_noise_1 = quantifyGradient(dIn1_orientation)
I_quantified_gradient_noise_2 = quantifyGradient(dIn2_orientation)

I_nms_noise_1 = nms(dIn1_norm, I_quantified_gradient_noise_1, threshold)
I_nms_noise_2 = nms(dIn2_norm, I_quantified_gradient_noise_2, threshold)

showImages([(I_nms_1,'Non-max suppression without noise'),
            (I_nms_noise_1,'Non-max suppression with noise = ' + str(sigma_noise_2)),
            (I_nms_noise_2,'Non-max suppression with noise = ' + str(sigma_noise_3))],
           'noise_non_max_suppresion',True)


# As we can see, the non-max suppression is robust towards noise because we blurred the image first.<br> 
# But, if the noise is too strong, we start to get some unwanted edges in the non-max suppresion.

# ### Canny edges

# In[15]:


from canny_edges import cannyEdges
sigma = 2
t1 = 0.07
t2 = 0.002
t3 = 0.03
t4 = t2

if file_format == 'jpg':
    t1*=255
    t2*=255
    t3*=255
    t4*=255
    
Edges_1 = cannyEdges(I,sigma,t1,t2)
I_nms_1 = nms(dI_norm, I_quantified_gradient, t1)
Edges_2 = cannyEdges(I,sigma,t3,t4)
I_nms_2 = nms(dI_norm, I_quantified_gradient, t3)

showImages([(I_nms_1,'Non-max suppresion for t1 = '+str(round(t1,2))),
            (Edges_1,'Canny edge detection for t1 = '+str(round(t1,2))+' and t2 = '+str(round(t2,2))),
            (I_nms_2,'Non-max suppresion for t3 = '+str(round(t3,2))),
            (Edges_2,'Canny edge detection for t3 = '+str(round(t3,2))+' and t4 = '+str(round(t4,2)))]
           ,'canny_edge',True)


# **Remember that if we are on a jpg file, we need a threshold which is 255 times the same threshold needed for a PNG file.**
# In conclusion, when the highest threshold is lowered, we get more details but soon enough we start to get some edges we don't want. 

# ### Canny edges detector with other images

# In[16]:


dpi = 60
im_data = plt.imread('results/lena_canny_edge.jpg')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
imgplot = plt.imshow(im_data)
plt.show()


# The next two images are PNG images. As said before, for the PNG images, the threshold needs to be divided by 255 compared to the same threshold for a JPG image. 

# In[17]:


dpi = 60
im_data = plt.imread('results/flower_canny_edge.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
imgplot = plt.imshow(im_data)
plt.show()


# In[18]:


dpi = 60
im_data = plt.imread('results/dome_canny_edge.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
imgplot = plt.imshow(im_data)
plt.show()


# ## III) Bilateral Filter 

# For the bilateral filter, I used the matlab function provided in **tp1_matlab/BF.m**.

# We choose range_sampling so that sigma_s = 2range_sampling.<br>
# We choose spatial_sampling so that sigma_r = 2spatial_sampling.

# In[19]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_1.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('sigma_spatial = 5 / sigma_sampling = 0.1')
imgplot = plt.imshow(im_data)
plt.show()


# In[20]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_2.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('sigma_spatial = 5 / sigma_sampling = 1')
imgplot = plt.imshow(im_data)
plt.show()


# In[21]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_3.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('sigma_spatial = 2.5 / sigma_sampling = 0.1')
imgplot = plt.imshow(im_data)
plt.show()


# In[22]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_4.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('sigma_spatial = 30 / sigma_sampling = 0.1')
imgplot = plt.imshow(im_data)
plt.show()


# In[23]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_5.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('sigma_spatial = 5 / sigma_sampling = 0.01')
imgplot = plt.imshow(im_data)
plt.show()


# In[24]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_6.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('sigma_spatial = 2 / sigma_sampling = 0.5')
imgplot = plt.imshow(im_data)
plt.show()


# **With all these experiments, we can conclude :**<br>
# Basically, sigma_spatial controls the sharpness of the edge. The closer it is to 2, the sharper are the edges. But if we go further than 2, it becomes too complicated for the computer to calculate the edges.<br>
# Sigma_sampling, on the other hand, controls the blurriness. Therefore, the higher it is, the blurrier the image will be.<br> 
# Therefore, there need to be a balance between sigma_spatial (low but too low) and sigma_sampling (high but not too high). It seems, for this image that **sigma_spatial = 2.5 and sigma_sampling = 0.1 yields the best solution.**

# In[25]:


dpi = 60
im_data = plt.imread('results/bilateral_filter_3.png')
height, width, dks = im_data.shape

figsize = width / float(dpi), height / float(dpi)

fig = plt.figure(figsize=figsize)
plt.axis('off')
plt.title('Best solution in all of the examples above : sigma_spatial = 2.5 / sigma_sampling = 0.1')
imgplot = plt.imshow(im_data)
plt.show()

