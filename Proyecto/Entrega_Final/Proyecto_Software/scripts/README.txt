---------------------------------
Escrito por Sergio Daniel Hernandez Charpak y Jose Francisco Molano
---------------------------------
Scripts creados para el analisis de las imagenes de corales tomadas por Nancy Ruiz
para el grupo BIOMMAR bajo la direccion de Susana Simancas en Uniandes
en el marco del proyecto del curso Imagenes y Vision dictado por Marcela Hernandez en el semestre 2016-10
---------------------------------
Instrucciones  -  Script Coral_Area_Measurement
---------------------------------
Script Coral_Area_Measurement
Script para el analisis individual de una imagen.
Para ejecutar el script se debe seguir la siguiente guia.
python Coral_Area_Measurement.py input_image_path images_extention(ex: JPG) output_folder_path intermediate_images_option(yes:1 no:0)
--------
Dependencias
-------------
numpy, matplotlib, pylab, sys, scikit-image
--------
Entrada
imagen a analizar.
La toma de la imagen debe ser tal que:
0. El fondo es claro
1. En la parte superior central se encuentra un cuadrado negro de 1cm2 de area
2. El coral se encuentra en la parte inferior central.
3. Ni el coral ni el cuadrado se encuentran pegados en el borde.
--------
Salida
1. imagen umbralizada con el coral y el cuadrado
2. archivo .dat con los resultados de los pasos intermediarios en el analisis
3. archivo .dat con el area en cm2 del coral.
--------
Argumento 1, input_image_path
Se trata de la ruta hacia la imagen a analizar.
Puede ser la ruta absoluta, o puede ser la ruta relativa
--------
Argumento 2, images_extention
Extension de la imagen a analizar.
Si se trata de una imagen .JPG ingresar aca JPG (Sin el punto)
--------
Argumento 3, output_folder_path
Se trata de la ruta hacia la carpeta donde se desea que esten los resultados.
La carpeta debe ser creada con anterioridad
Puede ser la ruta absoluta, o puede ser la ruta relativa
--------
Argumento 4, intermediate_images_option
Se trata de una opcion para que se guarden todas las imagenes intermediarias.
Si se desea que se creen y se guarden dichas imagenes, ingresar 1
Si no, ingresar 0 (u otro numero que no sea 1)
------------------------------------------------------------------
************************************
------------------------------------------------------------------
---------------------------------
Instrucciones  -  Script Coral_Area_Folder_Measurement
---------------------------------
Script Coral_Area_Folder_Measurement
Script para el analisis de las imagenes de una carpeta
Para ejecutar el script se debe seguir la siguiente guia.
python Coral_Area_Folder_Measurement.py input_folder_path images_extention(ex: JPG) output_folder_path
--------
Dependencias
-------------
numpy, matplotlib, pylab, sys, glob, scikit-image
--------
Entrada
carpeta con las imagenes a analizar.
La toma de la imagen debe ser tal que:
0. El fondo es claro
1. En la parte superior central se encuentra un cuadrado negro de 1cm2 de area
2. El coral se encuentra en la parte inferior central.
3. Ni el coral ni el cuadrado se encuentran pegados en el borde.
--------
Salida
1. imagenes umbralizadas con el coral y el cuadrado
2. archivo .dat por imagen con los resultados de los pasos intermediarios en el analisis
3. archivo .dat para la carpeta con el area en cm2 de los corales.
--------
Argumento 1, input_folder_path
Se trata de la ruta hacia la carpeta a analizar.
La carpeta contiene las imagenes
Puede ser la ruta absoluta, o puede ser la ruta relativa
--------
Argumento 2, images_extention
Extension de las imagenes a analizar.
Si se trata de imagenes .JPG ingresar aca JPG (Sin el punto)
--------
Argumento 3, output_folder_path
Se trata de la ruta hacia la carpeta donde se desea que esten los resultados.
La carpeta debe ser creada con anterioridad
Puede ser la ruta absoluta, o puede ser la ruta relativa
