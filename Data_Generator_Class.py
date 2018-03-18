import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import to_categorical

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, data, database_size, img_height, img_width, num_classes, num_input_channels, batch_size = 32, shuffle = True, data_augmentation = False):
      'Initialization'
      self.data = data
      self.database_size = database_size
      self.img_height =img_height
      self.img_width = img_width
      self.num_classes = num_classes
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.data_augmentation = data_augmentation
      self.num_input_channels = num_input_channels

  def frequency_balancing(self):
      'Median frequency balacing'
      # Initialize class weights
      class_weights = []

      # Initialize class frequencies
      class_frequencies = []
      for j in range(self.num_classes):
          class_frequencies.append([])

      # Compute the number of pixels on the picture
      total_number_pixels = self.img_height * self.img_width

      # For every picture inthe dataset, compute every class frequency
      for i in range(self.database_size):
          Y = to_categorical(np.reshape(self.data['mask'][i], (self.img_height, self.img_width, 1)), num_classes=self.num_classes)
          for j in range(self.num_classes):
              class_frequencies[j].append(np.sum(Y[:,:,j]) / total_number_pixels)

      # Compute classes weights
      for j in range(self.num_classes):
          class_weights.append(np.median(class_frequencies) / np.mean(class_frequencies[j]))

      return class_weights

  def generate(self, label=False):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          list_IDs = np.arange(self.database_size)
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              if label:
                  X, Y = self.__label_generation(list_IDs_temp)
              else:
                  X, Y = self.__data_generation(list_IDs_temp)

              yield X, Y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.img_height, self.img_width, self.num_input_channels))
      Y = np.empty((self.batch_size, self.img_height, self.img_width, self.num_classes))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store image
          X[i, :, :, :] = self.data['image'][ID]

          # Store mask
          Y[i, :, :, :] = to_categorical(np.reshape(self.data['mask'][ID], (self.img_height, self.img_width, 1)), num_classes=2)

          if self.data_augmentation:
              middle = self.batch_size//2
              # rotate half of the batch
              aug = iaa.Affine(rotate=(-45, 45)) #rotate randomly between -45 and 45 degrees
              aug_det = aug.to_deterministic()
              X[:middle, :, :, :] = aug_det.augment_images(X[:middle, :, :, :])
              Y[:middle, :, :, :] = aug_det.augment_images(Y[:middle, :, :, :])
              # flip horizontally and vertically the other half
              seq = iaa.Sequential([
                  iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                  iaa.Flipud(0.5)  # verticallu flip 50% of the images
              ])
              seq_det = seq.to_deterministic()
              X[middle:, :, :, :] = seq_det.augment_images(X[middle:, :, :, :])
              Y[middle:, :, :, :] = seq_det.augment_images(Y[middle:, :, :, :])

      return X, Y

  def __label_generation(self, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.img_height, self.img_width, self.num_input_channels))
      Y = np.empty((self.batch_size, 1))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store image
          X[i, :, :, :] = self.data['image'][ID]
          # Store mask
          Y[i, :, :, :] = to_categorical(np.reshape(self.data['label'][ID], (-1, 1)), num_classes=self.num_classes)

      return X, Y
