import matplotlib, os, glob, fnmatch
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import DatasetHandler_ISPRS
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

learn_dsm = False

# DATA

dataset = DatasetHandler_ISPRS.DatasetHandler_ISPRS(load_only_partition = 0.2, resize_down_to=192) #, resize_down_to=192)
# 0.2 and 192 res is doable on this potato

all_x = dataset.tiles_img
all_y = dataset.tiles_label

if learn_dsm: # Would need to be checked first when loading I think
   all_y = dataset.tiles_dsm
   all_y = all_y.reshape(all_y.shape + (1,))

x_train, x_val, y_train, y_val = train_test_split(all_x, all_y, test_size=0.05, random_state=42) # have more to train, almost all of it actually

# y must also be one hot for the softmax to work

print("x_train:", len(x_train), x_train.shape)
print("x_val:", len(x_val), x_val.shape)
print("y_train:", len(y_train), y_train.shape)
print("y_val:", len(y_val), y_val.shape)

# MODEL

# try another network architectures like FPN and PSPNet (they are more suitable for multiclass segmentation problem).
BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet', classes=6, activation='softmax')
if learn_dsm:
   model = Unet(BACKBONE, encoder_weights='imagenet', activation='sigmoid')

model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

#model.summary()
EPOCHS = 20
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8, # 32 froze while doing some other stufffs
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
)

mode_name = "SemSeg"
special_name = "111only0.2dataset"
if learn_dsm:
   mode_name = "DSM"
model.save("model_UNet-Resnet34_" + mode_name + "_95percOfTrain_8batch_"+str(EPOCHS)+"ep"+special_name+".h5")


dataset.debugger.nice_plot_history(history, show=False, save=True, name="model_UNet-Resnet34_" + mode_name + "_95percOfTrain_8batch_"+str(EPOCHS)+"ep"+special_name+"")
