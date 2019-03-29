import cv2
import numpy as np
# use a trained model to predict something from a validation set!

import matplotlib, os, glob, fnmatch
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

### Version ##########################################################################################


load_from = "model_UNet-Resnet34_SemSeg_95percOfTrain_8batch_15ep111only0.2dataset.h5" # predicting labels, as softmax
#load_from = "model_UNet-Resnet34_SemSeg_95percOfTrain_8batch_100ep111only0.2dataset.h5"
#load_from = "model_UNet-Resnet34_SemSeg_95percOfTrain_8batch_20ep111only0.2dataset.h5" # hoping that this is before it overfits? NOPE
learn_dsm = False

### DATA ##########################################################################################

import DatasetHandler_ISPRS
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss # loaded model needs it!
from segmentation_models.metrics import iou_score
dataset = DatasetHandler_ISPRS.DatasetHandler_ISPRS(load_only_partition = 0.01, resize_down_to=192) # for debugger


#load these:
#x_val ~ (N, 192, 192, 3)

folder_source = "data/12frameloop/"
N = 20
fileend = "png"

assert folder_source[-1] == "/"
from os import listdir

from keras.models import load_model
model = load_model(load_from)


for I in range(0,N,20):
   a = I
   b = I + 20
   print(a, "to", b)
   images_paths = [folder_source+f for f in listdir(folder_source) if fileend in f]
   images_names = [f for f in listdir(folder_source) if fileend in f]

   b = min(b,len(images_paths))
   images_paths = images_paths[a:b] # subset it alright?
   images_names = images_names[a:b]
   #print(images_paths[0:5])

   images = [dataset.load_raster_image(p) for p in images_paths]
   images = np.asarray(images)
   print("loaded shape:", images.shape)
   x_val = images

   ### MODEL ##########################################################################################
   x_val_pred = model.predict(x_val)

   if learn_dsm: # chop of the last channel which keras needed, for the sake of vis.
      x_val_pred = x_val_pred[:,:,:,0]
   else:
      label2color = {}
      label2color[0] = [255, 255, 255]  # Impervious surfaces = White
      label2color[1] = [0, 0, 255]  # Building = Blue
      label2color[2] = [0, 255, 255]  # Low vegetation = LightBlue
      label2color[3] = [0, 255, 0]  # Tree = Green
      label2color[4] = [255, 255, 0]  # Car = Yellow
      label2color[5] = [255, 0, 0]  # Clutter/background = Red

      for i, image_softmax in enumerate(x_val_pred):
         print("image_softmax:", image_softmax.shape) # (1024, 1024, 6)
         image_labels = np.argmax(image_softmax, axis=2)
         print("image_labels:", image_labels.shape)  # (1024, 1024, 6)
         image = np.zeros((1024,1024,3))
         for ai in range(len(image)):
            for bi in range(len(image[0])):
               image[ai,bi,:] = label2color[image_labels[ai,bi]]

         #image = image.astype(int) nope
         image = image / 255.0

         #print("image:", image.shape)
         #import matplotlib.pyplot as plt
         #plt.figure()
         #plt.imshow(image)
         #plt.show()

         #print("image:", dataset.debugger.dynamicRangeInImage(image))

         name = "predicted_semseg/sem_" + str(i).zfill(2) + ".png"
         name = "predicted_semseg/sem_" + images_names[i]
         dataset.save_raster_image(image, name)

   del x_val
   del x_val_pred
