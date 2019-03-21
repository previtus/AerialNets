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


load_from = "last_model_saved_8batch_100ep.h5" # predicting labels, as softmax
learn_dsm = False

load_from = "last_model_saved_8batch_50ep_dsm01ihope.h5" # predicting dsm as sigmoid, in range of 0-1 (roughly, irl up to 0.5?)
learn_dsm = True

load_from = "model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5"

### DATA ##########################################################################################

import DatasetHandler
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss # loaded model needs it!
from segmentation_models.metrics import iou_score
dataset = DatasetHandler.DatasetHandler(load_only_partition = 0.01) # for debugger

#load these:
#x_val ~ (N, 256, 256, 3)

#folder_source = "data/02_LongInterpolationOut500N30Fps_1024px_10200modelWith512/images_LongInterpolationOut500N30Fps/"
#N = 15000
#fileend = "jpg"

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


   #from keras.models import load_model
   #model = load_model(load_from)
   #model.summary()

   x_val_pred = model.predict(x_val)

   if learn_dsm: # chop of the last channel which keras needed, for the sake of vis.
      x_val_pred = x_val_pred[:,:,:,0]

   #x_val_pred = x_val_pred * 255


   print("pred", x_val_pred.shape)
   #print("pred ranges", dataset.debugger.dynamicRangeInImage(x_val_pred))

   from tqdm import tqdm
   for i in tqdm(range(len(x_val_pred))):
      name = "predicted_dsm/dsm_"+str(i).zfill(2)+".png"
      name = "predicted_dsm/dsm_"+images_names[i]
      dataset.save_raster_image(x_val_pred[i], name)

   del x_val
   del x_val_pred
#off = 0
#while off < len(x_val_pred):
#    dataset.debugger.viewTripples(x_val, x_val, x_val_pred, how_many=2, off=off)
#    #dataset.debugger.viewQuadrupples(x_val, x_val, x_val_pred, x_val_pred, how_many=4, off=off)
#    off += 2
