from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import load_model, Model
import keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
########################################################################

img_width, img_height = 224, 224

# train_data_dir ='/Users/akshen/Programs/Python/baby-monitoring-camera-detector/dataset/train'
# test_data_dir ='/Users/akshen/Programs/Python/baby-monitoring-camera-detector/dataset/test'

# epochs = 10
# batch_size = 2
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

# train_datagen = ImageDataGenerator(
#             rescale= 1. /255,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(train_data_dir,
#                                                     target_size=(img_width,img_height),
#                                                     batch_size=batch_size,
#                                                     class_mode='binary')

# test_generator = test_datagen.flow_from_directory(test_data_dir,
#                                                   target_size=(img_width,
#                                                                img_height),
#                                                   batch_size=batch_size,
#                                                   class_mode='binary')

########################################################################

# vgg16_model = VGG16()

# model = Sequential()
# for layer in vgg16_model.layers[:-1]:
#     model.add(layer)

# for layer in model.layers:
#     layer.trainable = False

# model.add(Dense(2, activation='softmax'))
# model.compile(Adam(lr=.001), loss='binary_crossentropy', metrics=['accuracy'])

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='give path',
#     monitor='val_acc',
#     mode='max',
#     save_best_only=True)

# model.fit(train_generator,
#                     validation_data=test_generator,
#                     epochs=epochs,
#                     steps_per_epoch=20,
#                     validation_steps=10, verbose=2, callbacks=[model_checkpoint_callback])
# model.save('testing_vgg.h5')
# print('Done............')

##########################################################################
model = load_model('testing_vgg.h5', compile=True)
#model.summary()
#print(model.history)
########################################################################
# img_pred = image.load_img('/Users/akshen/Programs/Python/baby-monitoring-camera-detector/dataset/train/in/24.jpg', target_size=(img_width,img_height))
# img_pred = image.img_to_array(img_pred)
# img_pred = np.expand_dims(img_pred, axis=0)

# result = model.predict(img_pred)
# if result[0][0] > result[0][1]:
#     print('In')
# else:
#     print('Out')
