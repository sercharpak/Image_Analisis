---------------------------------
Escrito por Sergio Daniel Hernandez Charpak y Jose Francisco Molano
---------------------------------
Notebooks creados para el analisis de las imagenes de corales tomadas por Nancy Ruiz
para el grupo BIOMMAR bajo la direccion de Susana Simancas en Uniandes
en el marco del proyecto del curso Imagenes y Vision dictado por Marcela Hernandez en el semestre 2016-10
---------------------------------
Instrucciones  -  Notebook Coral_Area_Measurement
---------------------------------
Notebook para el analisis individual de una imagen.
Para abrir correctamente el Notebook y hacer uso de este se debe tener instalado Ipython con notebook.
-------------
Dependencias
-------------
numpy, matplotlib, pylab, scikit-image
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
Variables de entrada
--------
Variable 1, input_image
Se trata de la ruta hacia la imagen a analizar.
Puede ser la ruta absoluta, o puede ser la ruta relativa
--------
Variable 2, images_extention
Extension de la imagen a analizar.
Si se trata de una imagen .JPG ingresar aca .JPG (Con el punto)
--------
Variable 3, outputfolder
Se trata de la ruta hacia la carpeta donde se desea que esten los resultados.
La carpeta debe ser creada con anterioridad
Puede ser la ruta absoluta, o puede ser la ruta relativa
--------
Variable 4, option_save_intermediate_images
Se trata de una opcion para que se guarden todas las imagenes intermediarias.
Si se desea que se creen y se guarden dichas imagenes, ingresar 1
Si no, ingresar 0 (u otro numero que no sea 1)
------------------------------------------------------------------
************************************
------------------------------------------------------------------
---------------------------------
Instrucciones  - Notebook Coral_Area_Folder_Measurement
---------------------------------
Notebook Coral_Area_Folder_Measurement
Notebook para el analisis de las imagenes de una carpeta
Para abrir correctamente el Notebook y hacer uso de este se debe tener instalado Ipython con notebook.
-------------
Dependencias
-------------
numpy, matplotlib, pylab, glob, scikit-image
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
Variables de entrada
--------
Variable 1, inputfolder
Se trata de la ruta hacia la carpeta a analizar.
La carpeta contiene las imagenes
Puede ser la ruta absoluta, o puede ser la ruta relativa
--------
Variable 2, images_extention
Extension de las imagenes a analizar.
Si se trata de imagenes .JPG ingresar aca .JPG (Con el punto)
--------
Variable 3, outputfolder
Se trata de la ruta hacia la carpeta donde se desea que esten los resultados.
La carpeta debe ser creada con anterioridad
Puede ser la ruta absoluta, o puede ser la ruta relativa
