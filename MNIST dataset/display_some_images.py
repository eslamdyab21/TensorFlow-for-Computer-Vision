import matplotlib.pyplot as plt
import numpy as np

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
