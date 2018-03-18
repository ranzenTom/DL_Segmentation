from keras.models import Model
from keras.layers import Input, Activation
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, UpSampling2D
from segmentationLayers import MaxPoolingWithArgmax2D, UpSamplingWithArgmax2D
from segmentationMetrics import mean_IU, pixel_accuracy, mean_accuracy
from segmentationLosses import weighted_categorical_crossentropy
from Data_Generator_Class import DataGenerator
import h5py
import pandas as pd

#---------------------------------------------------Model definition---------------------------------------------------
# Pictures dimensions
NB_CHANNELS_INPUTS = 3

# Number of classes
NUM_CLASSES = 2

# Input layer
inputs = Input(shape=(192, 192, NB_CHANNELS_INPUTS))

#First,let us build the encoder network
conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
[pool1, pool_indices1] = MaxPoolingWithArgmax2D(pool_size=(2, 2), padding="same")(conv1)

conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
[pool2, pool_indices2] = MaxPoolingWithArgmax2D(pool_size=(2, 2), padding="same")(conv2)

conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
[pool3, pool_indices3] = MaxPoolingWithArgmax2D(pool_size=(2, 2), padding="same")(conv3)

conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
[pool4, pool_indices4] = MaxPoolingWithArgmax2D(pool_size=(2, 2), padding="same")(conv4)

#Now, let us build the decoder
conv5 = UpSamplingWithArgmax2D()([pool4, pool_indices4])
conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)

conv6 = UpSamplingWithArgmax2D()([conv5, pool_indices3])
conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Activation('relu')(conv6)

conv7 = UpSamplingWithArgmax2D()([conv6, pool_indices2])
conv7 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('relu')(conv7)

conv8 = UpSamplingWithArgmax2D()([conv7, pool_indices1])
conv8 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Activation('relu')(conv8)

#Add the last layer to have an output with the input height and width but only two channels
conv9 = Conv2D(filters=NUM_CLASSES, kernel_size=(1,1), padding="same")(conv8)
#A sigmoid layer to have outputs between 0 and 1
predictions = Activation('softmax')(conv9)
#Finally, let us build the model
model = Model(inputs=inputs, outputs=predictions)


#------------------------------------------------------------Train model--------------------------------------------------

# Logs files
model_save_file = "outputs/SegNet_classweight_spacenet.h5"
record_csv_file = "outputs/SegNet_classweight_spacenet_records.csv"

# Parameters
training_data = h5py.File("Datasets/spacenet_phr_train.h5", "r")
training_database_size = training_data['image'].shape[0]
validation_data = h5py.File("Datasets/spacenet_phr_test.h5", "r")
validation_database_size = validation_data['image'].shape[0]
img_height = training_data['image'].shape[1]
img_width = training_data['image'].shape[2]
batch_size = 16
nb_epochs = 100
num_classes = NUM_CLASSES
num_input_channels = NB_CHANNELS_INPUTS
data_augment = True

# Create data generator objects
data_train_gen = DataGenerator(data=training_data, database_size=training_database_size, img_height=img_height, img_width=img_width, num_classes=num_classes, num_input_channels=num_input_channels, batch_size=batch_size, shuffle=True, data_augmentation=data_augment)
data_test_gen = DataGenerator(data=validation_data, database_size=validation_database_size, img_height=img_height, img_width=img_width, num_classes=num_classes, num_input_channels=num_input_channels, batch_size=batch_size, shuffle=True)

# Get class weights balancy based on the training db
class_weight = data_train_gen.frequency_balancing()
print("class_weight : ", class_weight)

# Create weighted categorical_crossentropy
weighted_loss = weighted_categorical_crossentropy(class_weight)

# Create the training and testing generators
training_generator = data_train_gen.generate()
validation_generator = data_test_gen.generate()

#Visualize model
model.summary()

# Compile model
model.compile(loss=weighted_loss, optimizer='adadelta', metrics=["accuracy", pixel_accuracy, mean_accuracy, mean_IU])

# Train model on dataset
pd.DataFrame(model.fit_generator(generator=training_generator, steps_per_epoch=training_database_size//batch_size, epochs=nb_epochs, validation_data=validation_generator, validation_steps=validation_database_size//batch_size).history).to_csv(record_csv_file)

print("The model has been succesfully trained on {} images then tested on {} images".format(training_database_size, validation_database_size))

# Save model weights
print("Save model")
model.save(model_save_file)


print("End of program :) !")
