import h5py
import os,sys
import numpy as np
import matplotlib.image as mpimg

#This program reads all the images .png of the spacenet_50cm database andwrite them into one single .h5 file """.

#Get pictures directory
#dir_path = "/home/thomas/Documents/Supaéro/3A/PIE/data/spacenet_8bits/8bits/50cm/"
dir_path = "/home/thomas/PycharmProjects/PIE/data_pie/dstl/PHR/test/"
#set final db name
database_file_name = "dstl_phr_test.h5"
#Set pictures standard height and width
img_height = 192
img_width = 192

#Set threshold above which the tile is considered a city
building_threshold = 0.1

#Get all pictures paths
files = os.listdir(dir_path)

#Seperate images and labels (ie masks) paths
images = []
masks= []
for file in files:  # a list of every file in dir_path
    if not 'mask' in file:
        str = file.split(sep='.')
        file_mask = str[0] + '_mask.' + str[1]
        if os.path.isfile(dir_path + file_mask):
            images.append(dir_path + file)
            masks.append(dir_path + file_mask)
        else:
            print(file + " is not taken into account!")


batch_size = 100
# decompose the database in chunks to not saturate the RAM
for i in range(0, len(images), batch_size):
    sub_images = images[i:i + batch_size]
    sub_masks = masks[i:i + batch_size]
    sub_images_arr = []
    sub_masks_arr = []
    sub_label_arr = []
    for k, img in enumerate(sub_images):
        img_arr = mpimg.imread(img)
        mask_arr = mpimg.imread(sub_masks[k])
	#transform mask of shape (height, width, 3) in shape (height, width)
        mask_arr = np.amax(mask_arr, axis=2)
	#compute ratio for labelling
        building_ratio = np.sum(mask_arr) / (img_width*img_height)
        if img_arr.shape == (img_height, img_width, 3) and mask_arr.shape == (img_height, img_width):
            sub_images_arr.append(img_arr)
            sub_masks_arr.append(mask_arr)
            if building_ratio >= building_threshold:
                sub_label_arr.append(1)
            else:
                sub_label_arr.append(0)

    sub_images_arr = np.array(sub_images_arr)
    sub_masks_arr = np.array(sub_masks_arr)
    sub_label_arr = np.array(sub_label_arr)

    if i == 0:
        with h5py.File(database_file_name, 'a') as f:
            print("create .h5 database")
            # Creating dataset to store features
            X_dset = f.create_dataset('image', sub_images_arr.shape, dtype='f', maxshape=(
            None, sub_images_arr.shape[1], sub_images_arr.shape[2], sub_images_arr.shape[3]))
            X_dset[:] = sub_images_arr
            # Creating dataset to store labels
            y_dset = f.create_dataset('mask', sub_masks_arr.shape, dtype='f',
                                      maxshape=(None, sub_masks_arr.shape[1], sub_masks_arr.shape[2]))
            y_dset[:] = sub_masks_arr
            L_dset = f.create_dataset('label', sub_label_arr.shape, dtype='f',
                                      maxshape=(None,))
            L_dset[:] = sub_label_arr
            f.close()
    else:
        with h5py.File(database_file_name, 'a') as f:
            print("Save {} th batch of {} tuples in database".format(i/batch_size + 1, batch_size))
            f['image'].resize((f['image'].shape[0] + sub_images_arr.shape[0]), axis=0)
            f['image'][-sub_images_arr.shape[0]:] = sub_images_arr
            f['mask'].resize((f['mask'].shape[0] + sub_masks_arr.shape[0]), axis=0)
            f['mask'][-sub_masks_arr.shape[0]:] = sub_masks_arr
            f['label'][-sub_label_arr.shape[0]:] = sub_label_arr
            f.close()

print("End of program :) !")
