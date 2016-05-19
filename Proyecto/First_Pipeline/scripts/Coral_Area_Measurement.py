
# coding: utf-8

# #Coral Area Measurements
# ##First Look
# ###Sergio Daniel Hernandez Charpak
# ###Jose Francisco Molano


import pylab
import numpy as np
import matplotlib.pyplot as plt



from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.measure import label
from skimage import measure
from skimage import morphology



inputfolder = '../../Fotos_prueba/'


# This is a test on one of the images. We will make use of the glob library to analyze all the images

# In[23]:

input_image = 'rsz_img_6780.jpg'
output_name = './measurements_intermediate_steps_'+input_image.strip('.jpg')+'.dat'


# In[24]:

fileout = open(output_name, 'w')


# In[25]:

image = pylab.imread(inputfolder+input_image)


# In[26]:

fileout.write("%s \n"%(inputfolder+input_image))


# <p>The image is an RGB image. We will transform it on a first basis to simplify the process.</p>
# <p> We follow the example: http://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_rgb.html#example-color-exposure-plot-adapt-rgb-py </p>

# In[27]:

from skimage.color import rgb2gray
img = rgb2gray(image)


# In[28]:

#print (img.shape)


# In[29]:

#print (img)


# In[14]:

img_mean = np.mean(img)
img_std = np.std(img)
img_max = np.max(img)
img_min = np.min(img)


# In[30]:

fileout.write("%f %f %f %f \n"%(img_mean, img_std, img_max, img_min))


# In[15]:

threshold = img_min + 2.0 *img_std


# In[31]:

fileout.write("%f \n"%(threshold))


# In[32]:

mask = img < threshold
img_thresholded = np.zeros(img.shape)
img_thresholded[mask] = 255


# In[33]:

#print (mask)


# In[34]:

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image RGB')

ax1.imshow(img)
ax1.set_title('Image grey')

ax2.imshow(img_thresholded)
ax2.set_title('Threshold with th = '+ str(threshold))

for ax in axes:
    ax.axis('off')
plt.savefig("imgs_first_look.png",format = 'png')
plt.show()


# In[35]:

fig = plt.figure(figsize = (10,10))
plt.gray()
imshow(img_thresholded)
plt.savefig("img_man_umbr.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[36]:

plt.close(fig)


# Le cortamos los bordes a la imagen

# In[37]:

n_x, n_y = img_thresholded.shape


# In[38]:

fileout.write("%d %d \n"%(n_x, n_y))


# In[39]:

cut_prop = 0.02


# In[40]:

fileout.write("%f \n"%(cut_prop))


# In[41]:

n_x_new = int(n_x - n_x*cut_prop)
n_y_new = int(n_y - n_y*cut_prop)


# In[42]:

fileout.write("%d %d \n"%(n_x_new, n_y_new))


# In[43]:

img_thresholded_new = np.zeros((n_x_new,n_y_new))


# In[44]:

for i in range (n_x_new):
    for j in range(n_y_new):
        img_thresholded_new[i,j] = img_thresholded[int(n_x*cut_prop) + i, int(n_y*cut_prop) +j]


# In[45]:

fig = plt.figure(figsize = (10,10))
plt.gray()
imshow(img_thresholded_new)
plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[46]:

plt.close(fig)


# ##Square Area

# Now we get the area of the square

# In[51]:

threshold_up = int(n_y_new/4.0)


# In[56]:

fileout.write("%d \n"%(threshold_up))


# In[53]:

img_thresholded_up = np.zeros((n_x_new,n_y_new))


# In[54]:

for i in range (threshold_up):
    for j in range(n_y_new):
        img_thresholded_up[i,j] = img_thresholded_new[i,j]


# In[57]:

fig = plt.figure(figsize = (10,10))
plt.gray()
imshow(img_thresholded_up)
plt.savefig("img_man_umbr_cut_up.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[58]:

etiquetas, num = label(img_thresholded_up, connectivity=1, return_num=True)


# In[60]:

hist, bins_edges = np.histogram(etiquetas.ravel())


# In[67]:

etiquetas = morphology.remove_small_objects(etiquetas,hist[len(hist)-1])


# In[68]:

fig = plt.figure(figsize = (10,10))
plt.gray()
imshow(etiquetas)
#plt.savefig("img_man_umbr_cut_up.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[69]:

hist, bins_edges = np.histogram(etiquetas.ravel())


# In[71]:

print (hist)


# In[72]:

area_square = hist[len(hist)-1]


# In[73]:

fileout.write("%f \n"%(area_square))


# ## Coral Area

# We extract the connected components to get the coral area

# In[74]:

#blobs_labels = measure.label(img_thresholded_new, background=0)
etiquetas, num = label(img_thresholded_new, connectivity=1, return_num=True)


# In[75]:

fig = plt.figure(figsize = (10,10))
#plt.gray()
imshow(etiquetas)
#plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# Guardamos solo el objeto más grande luego del fondo

# In[76]:

hist, bins = np.histogram(etiquetas)


# In[77]:

x_hist = np.zeros(len(hist))
for i in range (len(x_hist)):
    x_hist[i] = (bins[i] + bins[i+1])/2.0


# In[102]:

plt.plot(x_hist,np.log10(hist))
plt.ylabel("$Log_{10}$ of # pixels")
plt.xlabel("Component Number")
#plt.xlim(1,len(hist))


# In[82]:

hist_sorted = np.sort(hist)


# In[88]:

threshold_rm_objects = hist_sorted[len(hist_sorted)-2]


# In[92]:

fileout.write("%f \n"%(threshold_rm_objects))


# In[90]:

c = morphology.remove_small_objects(etiquetas,threshold_rm_objects)


# In[93]:

fig = plt.figure(figsize = (10,10))
#plt.gray()
imshow(c)
plt.savefig("img_coral_holes_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[116]:

n_dilation_erosion = 6


# In[117]:

fileout.write("%d \n"%(n_dilation_erosion))


# In[118]:

d = c
for i in range (n_dilation_erosion):
       d = morphology.dilation(d)


# In[119]:

fig = plt.figure(figsize = (10,10))
#plt.gray()
imshow(d)
#plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[120]:

for i in range (n_dilation_erosion):
       d = morphology.erosion(d)


# In[82]:

fig = plt.figure(figsize = (10,10))
#plt.gray()
imshow(d)
#plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.show()


# In[121]:

etiquetas, num = label(d, connectivity=1, return_num=True)


# In[122]:

print (num)
print (etiquetas.shape)


# In[123]:

hist, bins = np.histogram(etiquetas.ravel())


# In[128]:

x_hist = np.zeros(len(hist))
for i in range (len(x_hist)):
    x_hist[i] = (bins[i] + bins[i+1])/2.0
plt.scatter(x_hist,hist)
plt.ylabel("$Log_{10}$ of # pixels")


# In[130]:

area_coral_pixels = (hist[len(hist)-1])


# In[131]:

fileout.write("%f \n"%(area_coral_pixels))


# In[132]:

area_coral_cm_2 = area_coral_pixels/area_square


# In[133]:

fileout.write("%f \n"%(area_coral_cm_2))


# In[134]:

fileout.close()


# In[135]:

output_name = './area_coral_cm2_'+input_image.strip('.jpg')+'.dat'
fileout = open(output_name, 'w')
fileout.write("%f \n"%(area_coral_cm_2))
fileout.close()

