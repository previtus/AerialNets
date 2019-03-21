
# use a trained model to predict something from a validation set!

import matplotlib, os, glob, fnmatch
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

# FORCE CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### Version ##########################################################################################


load_from = "last_model_saved_8batch_100ep.h5" # predicting labels, as softmax
learn_dsm = False

load_from = "last_model_saved_8batch_50ep_dsm01ihope.h5" # predicting dsm as sigmoid, in range of 0-1 (roughly, irl up to 0.5?)
learn_dsm = True


### DATA ##########################################################################################

import DatasetHandler
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score



# DATA

dataset = DatasetHandler.DatasetHandler(load_only_partition = 0.01) # partition <> 1.0 will break the random sampling

all_x = dataset.tiles_img
all_y = dataset.tiles_label

if learn_dsm:
   all_y = dataset.tiles_dsm
   all_y = all_y.reshape(all_y.shape + (1,))

x_train, x_val, y_train, y_val = train_test_split(all_x, all_y, test_size=0.33, random_state=42)

# y must also be one hot for the softmax to work

print("x_train:", len(x_train), x_train.shape)
print("x_val:", len(x_val), x_val.shape)
print("y_train:", len(y_train), y_train.shape)
print("y_val:", len(y_val), y_val.shape)

### MODEL ##########################################################################################


from keras.models import load_model
model = load_model(load_from)
model.summary()

x_val_pred = model.predict(x_val)

if learn_dsm: # chop of the last channel which keras needed, for the sake of vis.
   y_val = y_val[:,:,:,0]
   x_val_pred = x_val_pred[:,:,:,0]

print("gt", y_val.shape)
print("pred", x_val_pred.shape)

print("gt ranges", dataset.debugger.dynamicRangeInImage(y_val))
print("pred ranges", dataset.debugger.dynamicRangeInImage(x_val_pred))

off = 0
while off < len(x_val_pred):
    dataset.debugger.viewQuadrupples(x_val, x_val, y_val, x_val_pred, how_many=2, off=off)
    off += 2

