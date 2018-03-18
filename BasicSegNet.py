from keras.models import Model
from keras.layers import Input, Activation
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, UpSampling2D
import pandas as pd
import h5py
from Data_Generator_Class import DataGenerator
from segmentationMetrics import mean_IU, pixel_accuracy, mean_accuracy

#---------------------------------------------------Model definition---------------------------------------------------
# Pictures dimensions
NB_CHANNELS_INPUTS = 3

# Input layer
inputs = Input(shape=(None, None, NB_CHANNELS_INPUTS))

#First,let us build the encoder network
conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

conv2= Conv2D(filters=128, kernel_size=(3, 3), padding="same")(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)

conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv3)

conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(conv3)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)

#Now, let us build the decoder
conv5 = Conv2D(filters=512, kernel_size=(3,3), padding="same")(conv4)
conv5= BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)

conv6 = UpSampling2D(size=(2,2))(conv5)
conv6 = Conv2D(filters=256, kernel_size=(3,3), padding="same")(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Activation('relu')(conv6)

conv7 = UpSampling2D(size=(2,2))(conv6)
conv7 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('relu')(conv7)

conv8 = UpSampling2D(size=(2,2))(conv7)
conv8 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Activation('relu')(conv8)

#Add the last layer to have an output with the input height and width but only two channels
conv9 = Conv2D(filters=2, kernel_size=(1,1), padding="same")(conv8)
#A sigmoid layer to have outputs between 0 and 1
predictions = Activation('softmax')(conv9)
#Finally, let us build the model
model = Model(inputs=inputs, outputs=predictions)

#------------------------------------------------------------Train model--------------------------------------------------

# Logs files
model_save_file = "outputs/basicSegNet_jacmel_phr.h5"
record_csv_file = "outputs/basicSegNet_jacmel_phr_records.csv"

# Parameters
training_data = h5py.File("Datasets/jacmel_phr_train.h5", "r")
training_database_size = training_data['image'].shape[0]
validation_data = h5py.File("Datasets/jacmel_phr_test.h5", "r")
validation_database_size = validation_data['image'].shape[0]
img_height = training_data['image'].shape[1]
img_width = training_data['image'].shape[2]
batch_size = 16
nb_epochs = 25
data_augment = True

# Create the training generator
training_generator = DataGenerator(data=training_data, database_size=training_database_size, img_height=img_height, img_width=img_width, batch_size=batch_size, shuffle=True, data_augmentation=data_augment).generate()
validation_generator = DataGenerator(data=validation_data, database_size=validation_database_size, img_height=img_height, img_width=img_width, batch_size=batch_size, shuffle=True).generate()

#Visualize model
model.summary()
# Compile model
model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy", pixel_accuracy, mean_accuracy, mean_IU])

# Train model on dataset
pd.DataFrame(model.fit_generator(generator=training_generator, steps_per_epoch=training_database_size//batch_size, epochs=nb_epochs, validation_data=validation_generator, validation_steps=validation_database_size//batch_size).history).to_csv(record_csv_file)

print("The model has been succesfully trained on {} images then tested on {} images".format(training_database_size, validation_database_size))

# Save model weights
print("Save model")
model.save(model_save_file)


print("End of program :) !")