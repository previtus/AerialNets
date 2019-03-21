import matplotlib, os, glob, fnmatch
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

# FORCE CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import DatasetHandler
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

learn_dsm = True

# DATA

dataset = DatasetHandler.DatasetHandler()

all_x = dataset.tiles_img
all_y = dataset.tiles_label

if learn_dsm:
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
model = Unet(BACKBONE, encoder_weights='imagenet', classes=3, activation='softmax')
if learn_dsm:
   model = Unet(BACKBONE, encoder_weights='imagenet', activation='sigmoid')

model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

#model.summary()

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8, # 32 froze while doing some other stufffs
    epochs=100,
    validation_data=(x_val, y_val),
)

#print(history)
#print(history.history)
# {'val_loss': [3.5243832300294122, 0.7317463924299996], 'val_iou_score': [0.2210197071984129, 0.4413165140826747], 'loss': [1.0412537466699832, 0.7205597858562648], 'iou_score': [0.2809795074373762, 0.4251761650927713]}

special_name = ""
if learn_dsm:
   special_name = "_dsm01proper"
model.save("model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep"+special_name+".h5")

# ps: batch 4 was maybe too little?

dataset.debugger.nice_plot_history(history, show=False, save=True, name="model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep"+special_name+".h5")

