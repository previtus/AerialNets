import os
import numpy as np
import matplotlib.pyplot as plt
from Debugger import Debugger
from skimage import io
from tqdm import tqdm
from keras.utils import to_categorical
import re
from skimage.transform import resize


class DatasetHandler_ISPRS(object):
    """
    # 0 checking the dataset

    Dataset > http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html

    # Satellite imagery is available in equal sized tiles.
    # Each tile has a size of 6000x6000 pixels
    # each exists in variants: rgb, label, dsm
    
    # label info:
    - Impervious surfaces (RGB: 255, 255, 255)
    - Building (RGB: 0, 0, 255)
    - Low vegetation (RGB: 0, 255, 255)
    - Tree (RGB: 0, 255, 0)
    - Car (RGB: 255, 255, 0)
    - Clutter/background (RGB: 255, 0, 0)
    """

    def __init__(self, load_only_partition = 1.0, resize_down_to=192):

        # this is approx 19573 images, which don't fit into MEM in 256x256x3
        # with the resolution of 192x192 this would roughly eval to the same as what we worked with last time (11136, 256, 256, 3)
        if resize_down_to is not None:
            self.do_resize = True
            self.resize_to = resize_down_to
        else:
            self.do_resize = False


        self.debugger = Debugger(None)
        self.tiles_img, self.tiles_label, self.tiles_dsm = self.load_yourself(load_only_partition)



    def img2onehot(self,img, num_of_classes = 6):
        #print(img.shape)
        onehot = to_categorical(img, num_classes=num_of_classes) # can it always match right???
        #print(onehot.shape)
        return onehot

    # tile into 256x256 blocks
    def image2tiles(self, img, size=256):
        # in this case the input images come as 6000x6000,
        # we can split them into 23 256x256 tiles on one side (which is 5888px)
        # the last one even could be overlapping with the previous one ...
        if len(img.shape) > 2:
            w, h, _ = img.shape
        else:
            w, h = img.shape
        #assert w==h # one dsm has 5999 instead of 6000, but it doesn't really matter
        n_splits = int(np.floor(w / size)) # care in general case
        # floor(6000 / 256) = 23

        tiles = []
        for i in range(n_splits):
            for j in range(n_splits):
                x = i*size
                y = j*size
                if len(img.shape) > 2:
                    tile = img[x:x+size, y:y+size, :]
                else:
                    tile = img[x:x + size, y:y + size]

                if self.do_resize:
                    tile = resize(tile, (self.resize_to,self.resize_to), anti_aliasing=True)
                    tile = np.asarray(tile)
                tiles.append(tile)
                #print(tile.shape)

        return tiles

    def load_raster_image(self, filename):
        img = io.imread(filename)
        arr = np.asarray(img)
        return arr

    def save_raster_image(self, array, name):
        io.imsave(name, array)

    def show_dsm2visible(self, dsm):
        dsm[dsm < -32760] = 0
        dsm = dsm + 30

        plt.figure()
        plt.imshow(dsm)

    def load_yourself(self, load_only_partition, overwrite_from_i=0, overwrite_to_i=-1):

        data_folder = "/home/pf/pfstaff/projects/ruzicka/DatasetTests/ISPRS_benchmark_datasets/Potsdam_extracts/"
        # 1_DSM_normalisation  2_Ortho_RGB  5_Labels_all
        rgb_folder = data_folder + "2_Ortho_RGB/"
        lab_folder = data_folder + "5_Labels_all/"
        dsm_folder = data_folder + "1_DSM_normalisation/"

        rgb_files = os.listdir(rgb_folder)
        lab_files = os.listdir(lab_folder)
        dsm_files = os.listdir(dsm_folder)


        def natural_sort(l): 
            convert = lambda text: int(text) if text.isdigit() else text.lower() 
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            return sorted(l, key = alphanum_key)

        rgb_files = natural_sort(rgb_files)
        lab_files = natural_sort(lab_files) 
        dsm_files = natural_sort(dsm_files) 

        DSMs = []
        RGBs = []
        GTLs = [] #labels

        for file in rgb_files:
           if ".tif" in file:
              RGBs.append(rgb_folder+file)
              #RGBs.append(file)

        for file in dsm_files:
           #if "_normalized_ownapproach.jpg" in file: # some files are empty!
           if "_normalized_lastools.jpg" in file: 
              DSMs.append(dsm_folder+file)
              #DSMs.append(file)

        for file in lab_files:
           if ".tif" in file:
              GTLs.append(lab_folder+file)
              #GTLs.append(file)

        # debug print(list(zip(RGBs,DSMs,GTLs)))

        print("RGBs:", len(RGBs), RGBs[0:3])
        print("DSMs:", len(DSMs), DSMs[0:3])
        print("GTLs:", len(GTLs), GTLs[0:3])

        color2label = {}
        color2label[255255255] = 0  # Impervious surfaces
        color2label[255] = 1  # Building
        color2label[255255] = 2  # Low vegetation
        color2label[255000] = 3  # Tree
        color2label[255255000] = 4  # Car
        color2label[252255000] = 4  # Car ~ lovely mess in the labels
        color2label[255000000] = 5  # Clutter/background

        label2color = {}
        label2color[0] = (255, 255, 255)  # Impervious surfaces
        label2color[1] = (0, 0, 255)  # Building
        label2color[2] = (0, 255, 255)  # Low vegetation
        label2color[3] = (0, 255, 0)  # Tree
        label2color[4] = (255, 255, 0)  # Car
        label2color[5] = (255, 0, 0)  # Clutter/background

        tiles_img = []
        tiles_label = np.asarray([])
        tiles_dsm = []

        number_to_load = int( float(len(RGBs)) * load_only_partition )

        for image_i in tqdm(range(number_to_load)):
        #for image_i in tqdm(range(10)):
            if image_i == 12:
                print("Skippining file as this one is broken!", lab_files[image_i])
                continue

            print("With files:", lab_files[image_i])

            img = self.load_raster_image(RGBs[image_i])
            label = self.load_raster_image(GTLs[image_i])
            dsm = self.load_raster_image(DSMs[image_i])

            #self.debugger.viewTripples([img],[label],[dsm],how_many=1)

            label_cat_long = np.int64(label)
            label_cat_long = 1000000*np.int64(label[:,:,0]) + 1000*np.int64(label[:,:,1]) + np.int64(label[:,:,2])

            #try:
            label_cat = np.vectorize(color2label.get)(label_cat_long)
            del label_cat_long

            # if we want to show tripples later ~~~ debug
            #label_cat = label

            """
            except:
               print("label_cat_long:", label_cat_long)
               print("unique values in label_cat_long", self.debugger.occurancesInImage(label_cat_long))

               print("label:", label)
               print(np.min(label_cat))
               print(np.max(label_cat))
            """

            #print(label[0,0:20])
            #print(label_cat[0,0:20])

            print("label_cat:", label_cat.shape) #, self.debugger.dynamicRangeInImage(label_cat))  # 6000x6000
            """
            try:
               print("label_cat range:", self.debugger.dynamicRangeInImage(label_cat))  # 6000x6000
            except:
               print("label_cat:", label_cat)
               print("label:", label)
               print(np.min(label_cat))
               print(np.max(label_cat))
            #debug print("unique colors", self.debugger.occurancesInImage(label_cat))
            """

            print("img:", img.shape, self.debugger.dynamicRangeInImage(img)) # 6000x6000x3
            #self.debugger.occurancesInImage(label)
            # it contains 0s and 255s - in the six labels (i think)
            print("dsm:", dsm.shape, self.debugger.dynamicRangeInImage(dsm)) # 6000x6000
            # CARE, one DSM is actually (6000, 5999)

            label = label_cat
            #debug self.debugger.occurancesInImage(label)

            """
            # transforms
            # moving dsm
            # All the DSM values fit into: -52.079  <=>  160.119 , average =  -17.505  and most +- (std)  8.228
            # sigmoid rather than softmax then in the model...
            dsm[dsm < -32760] = -17.505 # the average value in non moved dsm values (ignoring the - minint)
            dsm = dsm + 53
            # now the values go from 0 to 213
            dsm = dsm / 213.0
            # and finally 0-1
            """

            try:
                label = self.img2onehot(label) # when we need softmax
                #print(0)
            except:
                print("We had trouble!!!")
                # seems ok, no more strange labels ....
                self.debugger.occurancesInImage(label)
                nbkjbjkStopHalpEtc

            tile_img = self.image2tiles(img)
            tile_label = self.image2tiles(label)
            tile_dsm = self.image2tiles(dsm)

            #self.debugger.viewTripples(tile_img,tile_label,tile_dsm,how_many=3)

            tile_label_np = np.asarray(tile_label)

            #print(tiles_label.shape, tile_label_np.shape)

            tiles_img += tile_img
            if len(tiles_label) > 0:
                tiles_label = np.append(tiles_label, tile_label_np, axis=0)
            else:
                tiles_label = tile_label_np
            tiles_dsm += tile_dsm

            # memory is scarce ...
            del img
            del label
            del dsm
            del label_cat

        tiles_img = np.asarray(tiles_img)
        tiles_dsm = np.asarray(tiles_dsm)

        print("Loaded in total:")
        print("tiles_img:", len(tiles_img), tiles_img.shape)
        print("tiles_label:", len(tiles_label), tiles_label.shape)
        print("tiles_dsm:", len(tiles_dsm), tiles_dsm.shape)

        """
        # bonus analysis of the original values!
        all_dsm_values = tiles_dsm.flatten()

        min_val = np.round(np.nanmin(all_dsm_values), 3)
        max_val = np.round(np.nanmax(all_dsm_values), 3)
        avg_val = np.round(np.nanmean(all_dsm_values), 3)
        std_val = np.round(np.nanstd(all_dsm_values), 3)

        print("All the DSM values fit into:", min_val, " <=> ", max_val, ", average = ", avg_val, " and most +- (std) ", std_val)
        """


        return tiles_img, tiles_label, tiles_dsm



# testing here ...
#dataset = DatasetHandler_IPRS()

