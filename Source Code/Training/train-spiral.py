import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import cv2
import random
import sklearn.model_selection as model_selection
import datetime

from contextlib import redirect_stdout
categories = ["healthy", "parkinson"]
SIZE = 120

from tensorflow import keras

from tensorflow.keras import layers
import os


def createModel(train_data=None):
    if os.path.exists('model.h5') and train_data is None:
        try:
            print(__name__)
            model = keras.models.load_model('model.h5')
            print("returned")
            return model
        except Exception as e:
            print("error")


    elif train_data is not None:
        model = keras.Sequential([

            keras.Input(shape=train_data.shape[1:]),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),

            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax")

        ])
        return model



def getData():
    rawdata = []
    data = []
    dir = "D://FREELANCE PROJECTS//PARKINSON DISEASE DETECTION//DATASET//spiral"
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                rawdata = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_data = cv2.resize(rawdata, (SIZE, SIZE))

                data.append([new_data, class_num])
            except Exception as e:
                pass

    random.shuffle(data)

    img_data = []
    img_labels = []
    for features, label in data:
        img_data.append(features)
        img_labels.append(label)
    img_data = np.array(img_data).reshape(-1, SIZE, SIZE, 1)
    img_data = img_data / 255.0
    img_labels = np.array(img_labels)

    return img_data, img_labels



data, labels = getData()
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.20)

train_data, val_data, train_labels, val_labels = model_selection.train_test_split(train_data, train_labels,
                                                                                  test_size=0.10)
print(len(train_data), " ", len(train_labels), len(test_data), " ", len(test_labels))

model = createModel(train_data)

checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss',
                                             mode='min')

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"], )


history = model.fit(train_data, train_labels, epochs=100, validation_data=(val_data, val_labels)
                    )

model.save('model.h5')
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Model Accuracy: ", test_acc, "Model Loss: ", test_loss)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
