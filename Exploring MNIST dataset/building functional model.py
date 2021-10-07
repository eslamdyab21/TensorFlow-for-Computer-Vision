import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalMaxPool2D,GlobalAvgPool2D
from tensorflow.python.keras import activations




def functional_model():

    my_input = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model



def display_some_images(images,labels):
    plt.figure(figsize=(20,10))

    for i in range(25):
        idx = np.random.randint(0,images.shape[0])
        img = images[idx]
        label = labels[idx]

        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()


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

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    model.evaluate(x_test, y_test, batch_size=64)