to install label studio
###bash
pip install label-studio

to start label studio with enabling the local file sharing option
###bash
LOCAL_FILES_SERVING_ENABLED=true label-studio start

after labeling if you want to check the annotation and plot the annotations on the image use script 
###bash 
python3 visualize.py

this will generate annotated images in the folder for all the images in the directory



