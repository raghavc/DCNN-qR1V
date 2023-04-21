#imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import imghdr
import cv2
import imghdr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
directoryForPictures = 'Dataa' 
image_exts = ['jpeg','jpg', 'bmp', 'png']


for image_class in os.listdir(directoryForPictures): 
    for image in os.listdir(os.path.join(directoryForPictures, image_class)):
        image_path = os.path.join(directoryForPictures, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))


directoryForPictures = 'Dataa' 
image_exts = ['jpeg','jpg', 'bmp', 'png']


for image_class in os.listdir(directoryForPictures): 
    for image in os.listdir(os.path.join(directoryForPictures, image_class)):
        image_path = os.path.join(directoryForPictures, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)



os.listdir("Dataa")
shutil.rmtree("Dataa/.ipynb_checkpoints")

#building Data pipeline
data = tf.keras.utils.image_dataset_from_directory('Dataa')
#loop through Data pipeline
data_numpy_iterator = data.as_numpy_iterator()
#access data Pipeline
batch = data_numpy_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])




data = data.map(lambda x,y: (x/255, y))
#x is images
#y is target labels
#as we load data we scale using lambda
data.as_numpy_iterator().next()
#need to apply shuffling

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)




model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

img = cv2.imread('"NAMEOFTEST".jpg')
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5: 
    print(f'Predicted Image is Sober')
else:
    print(f'Predicted class is High')
