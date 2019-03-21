import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Debugger import Debugger
from skimage import io
from tqdm import tqdm
from keras.utils import to_categorical

class DatasetHandler(object):
    """
    # 0 checking the dataset

    Dataset > https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17007&compid=57607

    # Satellite imagery is available in equal sized tiles.
    # Each tile has a size of 2048x2048 pixels and a spatial resolution (ground sample distance) of ~0.5 meter.
    # For each tile the following 6 data files are available.
    # <title_id>_DSM.tif : the digital surface model of the tile; height in meters!
    #                      "-32767" we don't know, otherwise -30 ... +30 ?
    # <title_id>_RGB.tif : orthorectified RGB image. 3 bands of unsigned integers, 8 bits per band [0..255].
    # <title_id>_GTL.tif : class-level ground truth classification raster. Single band, 8 bit integer values [0..255], contains building / non-building / uncertain classification for each pixel, represented by values 6, 2, 65, respectively. See a note on uncertain values below.
    # <title_id>_GTC.tif : color image of class-level ground truth classification raster. Buildings are orange, uncertain regions are dark gray. This file does not contain more information than what is present in the _GTL files, it is provided only for visualization and convenience.

    # label info:
    # building = 6
    # non-building = 2
    # uncertain classification = 65
    """

    def __init__(self, load_only_partition = 1.0):
        a = 0
        self.debugger = Debugger(None)
        self.tiles_img, self.tiles_label, self.tiles_viz, self.tiles_dsm = self.load_yourself(load_only_partition)

        #tiles_img: 11136 (11136, 256, 256, 3)
        #tiles_label: 11136 (11136, 256, 256)
        #tiles_viz: 11136 (11136, 256, 256, 3)
        #tiles_dsm: 11136 (11136, 256, 256)


    def img2onehot(self,img):
        #print(img.shape)
        onehot = to_categorical(img, num_classes=3) # can it always match right???
        #print(onehot.shape)
        return onehot

    # tile into 256x256 blocks
    def image2tiles(self, img, size=256):
        # we have 8 splits, easy in this case
        if len(img.shape) > 2:
            w, h, _ = img.shape
        else:
            w, h = img.shape
        assert w==h
        n_splits = int(np.floor(w / size)) # care in general case

        tiles = []
        for i in range(n_splits):
            for j in range(n_splits):
                x = i*size
                y = j*size
                if len(img.shape) > 2:
                    tile = img[x:x+size, y:y+size, :]
                else:
                    tile = img[x:x + size, y:y + size]
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

    def load_yourself(self, load_only_partition):

        data_folder = "/PATHTOTopcoder-Urban-Mapper-3D/training/" # <<< CHANGE ACCORDINGLY !

        files = os.listdir(data_folder)

        DSMs = []
        RGBs = []
        GTLs = [] #labels
        GTCs = [] #visualizations

        for file in files:
            if "DSM" in file:
                DSMs.append(data_folder+file)
            elif "RGB" in file:
                RGBs.append(data_folder+file)
            if "GTL" in file:
                GTLs.append(data_folder+file)
            if "GTC" in file:
                GTCs.append(data_folder+file)

        print("RGBs:", len(RGBs), RGBs)
        print("DSMs:", len(DSMs), DSMs)
        print("GTLs:", len(GTLs), GTLs)
        print("GTCs:", len(GTCs), GTCs)

        tiles_img = []
        tiles_label = np.asarray([])
        tiles_viz = []
        tiles_dsm = []

        number_to_load = int( float(len(RGBs)) * load_only_partition )

        for image_i in tqdm(range(number_to_load)):

            img = self.load_raster_image(RGBs[image_i])
            label = self.load_raster_image(GTLs[image_i])
            viz = self.load_raster_image(GTCs[image_i])
            dsm = self.load_raster_image(DSMs[image_i])

            # transforms

            # building = 6 => 1
            # non-building = 2 => 0
            # uncertain classification = 65 => 2
            label[label == 6] = 1
            label[label == 2] = 0
            label[label == 65] = 2
            # some wild ones! ... make them all uncertain
            label[label == 17] = 2
            #self.debugger.occurancesInImage(tmp)

            # moving dsm
            # All the DSM values fit into: -52.079  <=>  160.119 , average =  -17.505  and most +- (std)  8.228
            # sigmoid rather than softmax then in the model...
            dsm[dsm < -32760] = -17.505 # the average value in non moved dsm values (ignoring the - minint)
            dsm = dsm + 53
            # now the values go from 0 to 213
            dsm = dsm / 213.0
            # and finally 0-1

            try:
                label = self.img2onehot(label) # when we need softmax
            except:
                print("We had trouble!!!")
                # seems ok, no more strange labels ....
                self.debugger.occurancesInImage(label)
                nbkjbjkStopHalpEtc

            tile_img = self.image2tiles(img)
            tile_label = self.image2tiles(label)
            tile_viz = self.image2tiles(viz)
            tile_dsm = self.image2tiles(dsm)

            tile_label_np = np.asarray(tile_label)
            #print("adding", len(tile_img))

            #print(tiles_label.shape, tile_label_np.shape)

            tiles_img += tile_img
            if len(tiles_label) > 0:
                tiles_label = np.append(tiles_label, tile_label_np, axis=0)
            else:
                tiles_label = tile_label_np
            tiles_viz += tile_viz
            tiles_dsm += tile_dsm

        tiles_img = np.asarray(tiles_img)
        tiles_viz = np.asarray(tiles_viz)
        tiles_dsm = np.asarray(tiles_dsm)

        print("Loaded in total:")
        print("tiles_img:", len(tiles_img), tiles_img.shape)
        print("tiles_label:", len(tiles_label), tiles_label.shape)
        print("tiles_viz:", len(tiles_viz), tiles_viz.shape)
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

        return tiles_img, tiles_label, tiles_viz, tiles_dsm
