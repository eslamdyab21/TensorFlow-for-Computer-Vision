import tensorflow
import numpy as np
from display_some_images import display_some_images
from deep_learning_functional_model import functional_model


if __name__=='__main__':

    (x_train,y_train), (x_test,y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train shape = ",x_train.shape)
    print("y_train shape = ", y_train.shape)
    print("x_test shape = ", x_test.shape)
    print("x_test shape = ", x_test.shape)

    if False:
        display_some_images(x_train,y_train)

    #Normalize
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32') / 255

    # add one extra dimention so that the Conv2D would work (it accepts 4d array[patsh, [img_gray_2d], num_of_chaells=1]
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print('')
    print('dimentions after adding extra dim:')
    print("x_train shape = ", x_train.shape)
    print("y_train shape = ", y_train.shape)
    print("x_test shape = ", x_test.shape)
    print("x_test shape = ", x_test.shape)

    model = functional_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')


    # model training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)
