import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import cv2 as cv
import numpy as num

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# Create Model
# model = Sequential()
#
# model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation=tf.nn.relu))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation=tf.nn.softmax))
#
# # Compile And Train
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x=x_train, y=y_train, epochs=5)
#
# Save Model
# loss, accuracy = model.evaluate(x_test, y_test)
# print(accuracy)
# print(loss)
# model.save('digits.model1')


# Test The Model
model1 = tf.keras.models.load_model('digits.model1')

letter = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


#
def prepare(path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)  # convert image to the single channel grayscale image
    image2 = cv.resize(image, (28, 28))
    image2 = num.invert(num.array([image2]))
    return image2.reshape(1, 28, 28, 1)


# image path
pre = model1.predict([prepare("C:\\Users\\FFaye\\OneDrive\\Desktop\\mid\\AI\\3.jpg")])
print(letter[num.argmax(pre)])
