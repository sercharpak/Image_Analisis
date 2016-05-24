#!/usr/bin/python

# #Coral Area Measurements
# ##First Look
# ###Sergio Daniel Hernandez Charpak
# ###Jose Francisco Molano

# Script creado para el analisis de las imagenes de corales tomadas por Nancy Ruiz
# para el grupo BIOMMAR bajo la direccion de Susana Simancas en Uniandes
# en el marco del proyecto del curso Imagenes y Vision dictado por Marcela Hernandez en el semestre 2016-10

# Usage
# python Coral_Area_Measurement.py input_image_path images_extention(ex: JPG) output_folder_path intermediate_images_option(yes:1 no:0)
USAGE = "python Coral_Area_Measurement.py input_image_path images_extention(ex: JPG) output_folder_path intermediate_images_option(yes:1 no:0)"

# ###Imports

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

# ###Functions

def get_folder_name(folder_path):
    folder_name_array = (folder_path.strip('/')).split('/')
    return folder_name_array[len(folder_name_array)-1]

def umbralizar_otsu(imagen):
    thresh = threshold_otsu(imagen)
    binary = imagen < thresh
    return binary, thresh

def guardar_imagen(image, path):
    fig = plt.figure(figsize = (10,10))
    plt.gray()
    plt.imshow(image)
    plt.savefig(path)
    plt.xlabel("y(pixels)")
    plt.ylabel("x(pixels)")
    #plt.show()
    plt.close(fig)

#---------------------------------------------------------
# Usage

if(len(sys.argv)!=5):
    print "Please use correctly"
    print USAGE
    sys.exit()

input_image = sys.argv[1]
images_extention = '.'+sys.argv[2]
outputfolder = sys.argv[3]
#If 1 is selected all the intermediate step images are saved
#If 0 is selected only
option_save_intermediate_images = int(sys.argv[4])

image_name = get_folder_name(input_image).strip(images_extention)

steps_filename = outputfolder + image_name+'_steps.dat'

file_steps_out = open(steps_filename, 'w')

image = pylab.imread(input_image)
file_steps_out.write("%s %s \n"%("img_path", input_image))

# <p>The image is an RGB image. We will transform it on a first basis to simplify the process.</p>
# <p> We follow the example: http://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_rgb.html#example-color-exposure-plot-adapt-rgb-py </p>

img = rgb2gray(image)

# ###Cortando la imagen

# Le cortamos los bordes a la imagen

n_x, n_y = img.shape

file_steps_out.write("%s %d %d \n"%("dim_x_y",n_x, n_y))

cut_prop = 0.04
cut_y_0 = 0.5
cut_y_final = 4.0

file_steps_out.write("%s %f \n"%("cut_proportion",cut_prop))
file_steps_out.write("%s %d %d \n"%("cut_y_0_final",cut_y_0, cut_y_final))

n_colums_x = int(n_x*cut_prop)
n_colums_y = int(n_y*cut_prop)

#Arrays of the column index to be deleted
array_index_columns_x_0 = np.arange(n_colums_x)
array_index_columns_y_0 = np.arange(n_colums_y + cut_y_0*n_colums_y)
array_index_columns_x_final = np.linspace(n_x-n_colums_x,n_x-1, n_colums_x)
array_index_columns_y_final = np.linspace(n_y-cut_y_final*n_colums_y,n_y-1, cut_y_final*n_colums_y)


img_delete_x = np.delete(img,array_index_columns_x_final,0 )
img_delete_x = np.delete(img_delete_x,array_index_columns_x_0,0 )
img_delete_y = np.delete(img_delete_x,array_index_columns_y_final,1 )
img_delete_y = np.delete(img_delete_y,array_index_columns_y_0,1 )


if (option_save_intermediate_images==1):
    guardar_imagen(img_delete_x, outputfolder+image_name+"_x_cut.png")

if (option_save_intermediate_images==1):
    guardar_imagen(img_delete_y, outputfolder+image_name+"_y_cut.png")

img = img_delete_y

n_x_new, n_y_new = img.shape

file_steps_out.write("%s %d %d \n"%("dim_new_x_y",n_x_new, n_y_new))

if (option_save_intermediate_images==1):
    fig = plt.figure(figsize = (10,10))
    bins=256
    plt.title("Cut Histogram")
    plt.xlabel("gray level")
    plt.ylabel("#(pixels)")
    plt.hist(img.ravel(), bins=bins, histtype='step', color='black')
    #plt.show()
    plt.savefig(outputfolder+image_name+"_histogram.png",format = 'png')
    plt.close(fig)

# ###Umbralization with Otsu


cut_y_img_umbr = 0.25 #Va a umbralizar el 0.25 de la imagen independientemente que el 0.75
n_y_cut = int(n_y_new*cut_y_img_umbr)

file_steps_out.write("%s %f \n"%("cut_otsu_thresholding",cut_y_img_umbr))

img_square = np.zeros((n_x_new, n_y_new))
img_coral = np.zeros((n_x_new, n_y_new))

for i in range (n_x_new):
    for j in range(n_y_new):
        if(j<n_y_cut):
            img_square[i,j] = img[i,j]
        else:
            img_coral[i,j] = img[i,j]

img_square_binary, thresh_square = umbralizar_otsu(img_square)
img_coral_binary, thresh_coral = umbralizar_otsu(img_coral)

file_steps_out.write("%s %f %f \n"%("Otsu_thresh_square_coral", thresh_square, thresh_coral))

if (option_save_intermediate_images==1):
    guardar_imagen(img_square_binary, outputfolder+image_name+"_square_otsu.png")

if (option_save_intermediate_images==1):
    guardar_imagen(img_coral_binary, outputfolder+image_name+"_coral_otsu.png")

img_thresholded = np.zeros((n_x_new, n_y_new))
for i in range (n_x_new):
    for j in range(n_y_new):
        if(j<n_y_cut):
            img_thresholded[i,j] = img_square_binary[i,j]
        else:
            img_thresholded[i,j] = img_coral_binary[i,j]

if (option_save_intermediate_images==1):
    fig, axes = plt.subplots(nrows=3, figsize=(14, 14))
    ax0, ax1, ax2 = axes
    plt.gray()
    ax0.imshow(image)
    ax0.set_title('Image RGB')
    ax1.imshow(img)
    ax1.set_title('Image grey')
    ax2.imshow(img_thresholded)
    ax2.set_title('Otsu Threshold with thresholds: \n square: '+ str(thresh_square) + '\n coral: '+ str(thresh_coral))
    for ax in axes:
        ax.axis('off')
    plt.savefig(outputfolder+image_name+"_first_look.png",format = 'png')
    #plt.show()
    plt.close(fig)

if (option_save_intermediate_images==1):
    guardar_imagen(img_thresholded, outputfolder+image_name+"_thresholded.png")

# ##Square Area
# Now we get the area of the square

threshold_up = int(n_y_new*cut_y_img_umbr)
file_steps_out.write("%s %d \n"%("square_cut", threshold_up))

img_thresholded_up = np.zeros((n_x_new,n_y_new))

for i in range (n_x_new):
    for j in range(threshold_up):
        img_thresholded_up[i,j] = img_thresholded[i,j]

if (option_save_intermediate_images==1):
    guardar_imagen(img_thresholded_up, outputfolder+image_name+"_thresholded_cut_up.png")

etiquetas_square, num = label(img_thresholded_up, connectivity=2, return_num=True)

hist, bins_edges = np.histogram(etiquetas_square.ravel())

etiquetas_square = morphology.remove_small_objects(etiquetas_square,np.sort(hist)[len(hist)-2] - 10)

if (option_save_intermediate_images==1):
    guardar_imagen(etiquetas_square, outputfolder+image_name+"_thresholded_cut_up_labels.png")

hist, bins_edges = np.histogram(etiquetas_square.ravel())

area_square = np.sort(hist)[len(hist)-2]

file_steps_out.write("%s %f \n"%("area_square_pixels", area_square))

etiquetas_square =(255/np.max(etiquetas_square))*etiquetas_square

image_scale = (255/np.max(img_thresholded))*img_thresholded

image_scale= image_scale - etiquetas_square

if (option_save_intermediate_images==1):
    guardar_imagen(image_scale, outputfolder+image_name+"_no_square.png")

# ## Coral Area
# We extract the connected components to get the coral area

etiquetas, num = label(image_scale, connectivity=2, return_num=True)

if (option_save_intermediate_images==1):
    guardar_imagen(etiquetas, outputfolder+image_name+"_no_square_labels.png")

# Guardamos solo el objeto mas grande luego del fondo

if (option_save_intermediate_images==1):
    etiquetas_scale = (255.0/np.max(etiquetas))*etiquetas
    fig = plt.figure(figsize = (10,10))
    bins=256
    plt.title("Histogram Coral Labels")
    plt.xlabel("gray level")
    plt.ylabel("# pixels")
    plt.hist(etiquetas_scale.ravel(), bins=bins, histtype='step', color='black')
    plt.savefig(outputfolder+image_name+"_histogram_coral_labels.png",format = 'png')
    #plt.show()
    plt.close(fig)

hist, bins = np.histogram(etiquetas.ravel())
hist_sorted = np.sort(hist)

threshold_rm_objects = int (hist_sorted[len(hist_sorted)-2] - hist_sorted[len(hist_sorted)-2]/5.0)

file_steps_out.write("%s %f \n"%("thresh_delet_small_objects", threshold_rm_objects))

c = morphology.remove_small_objects(etiquetas,threshold_rm_objects)

if (option_save_intermediate_images==1):
    c_scale = (255.0/np.max(c))*c
    fig = plt.figure(figsize = (10,10))
    bins=256
    plt.title("Histogram Coral Labels \n after removing small objects")
    plt.xlabel("gray level")
    plt.ylabel("# pixels")
    plt.hist(c_scale.ravel(), bins=bins, histtype='step', color='black')
    plt.savefig(outputfolder+image_name+"_histogram_coral_labels_no_small_objects.png",format = 'png')
    #plt.show()
    plt.close(fig)

if (option_save_intermediate_images==1):
    guardar_imagen(c, outputfolder+image_name+"_coral_with_holes.png")

n_dilation_erosion = 6

file_steps_out.write("%s %d \n"%("number_dilated_erode", n_dilation_erosion))

d = c
for i in range (n_dilation_erosion):
       d = morphology.dilation(d)

if (option_save_intermediate_images==1):
    guardar_imagen(d, outputfolder+image_name+"_coral_dilatated.png")

for i in range (n_dilation_erosion):
       d = morphology.erosion(d)

if (option_save_intermediate_images==1):
    guardar_imagen(d, outputfolder+image_name+"_coral_eroded.png")

hist, bins = np.histogram(d.ravel())

threshold_rm_objects =  (np.sort(hist)[len(hist)-2]) - int((np.sort(hist)[len(hist)-2])/50.0)

d = morphology.remove_small_objects(d,threshold_rm_objects)

etiquetas_coral, num = label(d, connectivity=2, return_num=True)

hist, bins = np.histogram(etiquetas_coral.ravel())

if (option_save_intermediate_images==1):
    etiquetas_coral_scale = (255.0/np.max(etiquetas_coral))*etiquetas_coral
    fig = plt.figure(figsize = (10,10))
    bins=256
    plt.title("Histogram Coral Labels \n after removing small objects \n and filling holes")
    plt.xlabel("gray level")
    plt.ylabel("# pixels")
    plt.hist(etiquetas_coral_scale.ravel(), bins=bins, histtype='step', color='black')
    plt.savefig(outputfolder+image_name+"_histogram_coral_labels_no_holes.png",format = 'png')
    #plt.show()
    plt.close(fig)

area_coral_pixels = np.sort(hist)[len(hist)-2]

file_steps_out.write("%s %f \n"%("area_coral_pixels", area_coral_pixels))

area_coral_cm_2 = area_coral_pixels/area_square

file_steps_out.write("%s %f \n"%("area_coral_cm2", area_coral_cm_2))
file_steps_out.close()

output_name = outputfolder + image_name+'_area_coral_cm2.dat'
fileout = open(output_name, 'w')
fileout.write("%s \t %f \n"%(image_name, area_coral_cm_2))
fileout.close()

etiquetas_coral = (255/np.max(etiquetas_coral))*etiquetas_coral
final_image = etiquetas_square + etiquetas_coral

guardar_imagen(final_image, outputfolder+image_name+"_final_image.png")
