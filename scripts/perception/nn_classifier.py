from tensorflow.python.keras.models import Sequential, Model, load_model

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, Dropout, Activation, MaxPooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import rmsprop
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import numpy as np


num_classes = 21


image_size = 64

data_generator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rescale=1. / 255,
        brightness_range=(0.5, 1.5), # 1 means original image
        samplewise_std_normalization = True,
        width_shift_range = 0.1,
        height_shift_range = 0.1)

train_generator = data_generator.flow_from_directory(
        './dataset/Training_Set',
        target_size=(image_size, image_size),
        batch_size= 32,
        class_mode='categorical',
        shuffle=True)


validation_generator = data_generator.flow_from_directory(
        './dataset/Test_Set',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_generator.image_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

for i,layer in enumerate(model.layers):
  print(i,layer.name)


# initiate RMSprop optimizer
opt = rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_val = validation_generator.n//validation_generator.batch_size

es = EarlyStopping(monitor='val_acc', mode='max', min_delta = 0.3, verbose=1, patience=3)
mc = ModelCheckpoint('model/SingleJules.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=30,
                   validation_data=validation_generator,
                   validation_steps=step_size_val,
                   callbacks=[es,mc])

model.save('model/SingleJules.h5')
