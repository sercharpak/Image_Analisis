# #Coral Area Measurements
# ##First Look
# ###Sergio Daniel Hernandez Charpak
# ###Jose Francisco Molano

import pylab
import numpy as np
import matplotlib.pyplot as plt
import sys

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.measure import label
from skimage import measure
from skimage import morphology
from skimage.color import rgb2gray

inputfolder = sys.argv[1]
input_image = sys.argv[2]

# This is a test on one of the images. We will make use of the glob library to analyze all the images

output_name = './measurements_intermediate_steps_'+input_image.strip('.jpg')+'.dat'

fileout = open(output_name, 'w')

image = pylab.imread(inputfolder+input_image)

fileout.write("%s \n"%(inputfolder+input_image))


# <p>The image is an RGB image. We will transform it on a first basis to simplify the process.</p>
# <p> We follow the example: http://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_rgb.html#example-color-exposure-plot-adapt-rgb-py </p>

img = rgb2gray(image)

img_mean = np.mean(img)
img_std = np.std(img)
img_max = np.max(img)
img_min = np.min(img)

fileout.write("%f %f %f %f \n"%(img_mean, img_std, img_max, img_min))

threshold = img_min + 2.0 *img_std

fileout.write("%f \n"%(threshold))

mask = img < threshold
img_thresholded = np.zeros(img.shape)
img_thresholded[mask] = 255

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

fig = plt.figure(figsize = (10,10))
plt.gray()
plt.imshow(img_thresholded)
plt.savefig("img_man_umbr.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")

plt.close(fig)

# Le cortamos los bordes a la imagen

n_x, n_y = img_thresholded.shape

fileout.write("%d %d \n"%(n_x, n_y))

cut_prop = 0.02

fileout.write("%f \n"%(cut_prop))

n_x_new = int(n_x - n_x*cut_prop)
n_y_new = int(n_y - n_y*cut_prop)

fileout.write("%d %d \n"%(n_x_new, n_y_new))

img_thresholded_new = np.zeros((n_x_new,n_y_new))

for i in range (n_x_new):
    for j in range(n_y_new):
        img_thresholded_new[i,j] = img_thresholded[int(n_x*cut_prop) + i, int(n_y*cut_prop) +j]

fig = plt.figure(figsize = (10,10))
plt.gray()
plt.imshow(img_thresholded_new)
plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)

# ##Square Area

# Now we get the area of the square

threshold_up = int(n_y_new/4.0)

fileout.write("%d \n"%(threshold_up))

img_thresholded_up = np.zeros((n_x_new,n_y_new))

for i in range (threshold_up):
    for j in range(n_y_new):
        img_thresholded_up[i,j] = img_thresholded_new[i,j]

fig = plt.figure(figsize = (10,10))
plt.gray()
plt.imshow(img_thresholded_up)
plt.savefig("img_man_umbr_cut_up.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)

etiquetas, num = label(img_thresholded_up, connectivity=1, return_num=True)

hist, bins_edges = np.histogram(etiquetas.ravel())

etiquetas_square = morphology.remove_small_objects(etiquetas,hist[len(hist)-1])
"""
fig = plt.figure(figsize = (10,10))
plt.gray()
plt.imshow(etiquetas)
#plt.savefig("img_man_umbr_cut_up.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)
"""
hist, bins_edges = np.histogram(etiquetas_square.ravel())

area_square = hist[len(hist)-1]

fileout.write("%f \n"%(area_square))

# ## Coral Area

# We extract the connected components to get the coral area

etiquetas, num = label(img_thresholded_new, connectivity=1, return_num=True)
"""
fig = plt.figure(figsize = (10,10))
plt.imshow(etiquetas)
#plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)
"""
# Guardamos solo el objeto más grande luego del fondo

hist, bins = np.histogram(etiquetas)
"""
x_hist = np.zeros(len(hist))
for i in range (len(x_hist)):
    x_hist[i] = (bins[i] + bins[i+1])/2.0
plt.plot(x_hist,np.log10(hist))
plt.ylabel("$Log_{10}$ of # pixels")
plt.xlabel("Component Number")
plt.close(fig)
"""
hist_sorted = np.sort(hist)

threshold_rm_objects = hist_sorted[len(hist_sorted)-2]

fileout.write("%f \n"%(threshold_rm_objects))

c = morphology.remove_small_objects(etiquetas,threshold_rm_objects)
"""
fig = plt.figure(figsize = (10,10))
plt.imshow(c)
plt.savefig("img_coral_holes_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)
"""
n_dilation_erosion = 6

fileout.write("%d \n"%(n_dilation_erosion))

d = c
for i in range (n_dilation_erosion):
       d = morphology.dilation(d)
"""
fig = plt.figure(figsize = (10,10))
plt.imshow(d)
#plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)
"""
for i in range (n_dilation_erosion):
       d = morphology.erosion(d)
"""
fig = plt.figure(figsize = (10,10))
plt.imshow(d)
#plt.savefig("img_man_umbr_cut.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)
"""
etiquetas_coral, num = label(d, connectivity=1, return_num=True)

hist, bins = np.histogram(etiquetas_coral.ravel())
"""
x_hist = np.zeros(len(hist))
for i in range (len(x_hist)):
    x_hist[i] = (bins[i] + bins[i+1])/2.0
plt.scatter(x_hist,hist)
plt.ylabel("$Log_{10}$ of # pixels")
plt.close(fig)
"""
area_coral_pixels = (hist[len(hist)-1])


fileout.write("%f \n"%(area_coral_pixels))

area_coral_cm_2 = area_coral_pixels/area_square

fileout.write("%f \n"%(area_coral_cm_2))

fileout.close()

output_name = './area_coral_cm2_'+input_image.strip('.jpg')+'.dat'
fileout = open(output_name, 'w')
fileout.write("%s %f \n"%(input_image, area_coral_cm_2))
fileout.close()

final_image = np.zeros((n_x_new,n_y_new))

for i in range(n_x_new):
	for j in range(n_y_new):
		final_image[i,j] = etiquetas_square[i,j] + etiquetas_coral[i,j]

fig = plt.figure(figsize = (10,10))
plt.gray()
plt.imshow(final_image)
plt.savefig("final_image.png",format = 'png')
plt.xlabel("y(pixels)")
plt.ylabel("x(pixels)")
plt.close(fig)
