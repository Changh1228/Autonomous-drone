#!/usr/bin/env python
from __future__ import print_function

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, Dropout, Activation, MaxPooling2D


LABEL_RECT = {0 : "airport", 1 : "residential", 2 : "CRAP"}
LABEL_TRI = {0 : "dangerous_curve_left", 1 : "dangerous_curve_right", 2 : "junction", 3 : "road_narrows_from_left",
                   4 : "road_narrows_from_right", 5 : "roundabout_warning", 6 : "CRAP"}
LABEL_CERCL = {0 : "follow_left", 1 : "follow_right", 2 : "no_bicycle", 3 : "no_heavy_truck",
                   4 : "no_parking", 5 : "no_stopping_and_parking", 6 : "stop"}

class NNClassifier:
    """
    NN to classify the 3 different kind of signs (rectangles, circles and triangles).
    One NN for each sign is trained.
    """
    def __init__(self):

        # Import my model
        self.model_rectangle = self.create_model(3)
        self.model_rectangle.load_weights("./model/model_rectangle_nou.h5")
        #-----------------------------
        self.model_triangle = self.create_model(7)
        self.model_triangle.load_weights("./model/model_triangle_my_data.h5")
        #-----------------------------
        self.model_cercle = self.create_model(7)
        self.model_cercle.load_weights("./model/model_cercle_nou.h5")

    def create_model(self, number_classes):
        """Creates the NN structure, number_classes means the number of possible outputs
        the multiclass problem has. There are 7 for triangle and cercle and 3 for rectangle
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(64, 64, 3)))
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
        model.add(Dense(number_classes))
        model.add(Activation('softmax'))
        return model
        
    def classify_nn(self, type, candidate, flag=5):
        """
        :param type: String stating kind of sign. Possible values: 'rectangle','triangle','circle'
        :param candidate: The wraped image of the sign after being cropped from the original image and after
        all the normalization (of mean and std)
        :return: the name of the sign, i.e: dangerous_curve_left
        """
        if type == "rectangle":
            classes = self.model_rectangle.predict_classes(candidate)
            return LABEL_RECT[classes[0]]
        elif type == "triangle":
            classes = self.model_triangle.predict_classes(candidate)
            return LABEL_TRI[classes[0]]
        else:
            classes = self.model_cercle.predict_classes(candidate)
            return LABEL_CERCL[classes[0]]

