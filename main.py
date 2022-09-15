import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
import scipy

dataset_name = 'oxford_flowers102'

train_dataset = tfds.load(dataset_name, split = tfds.Split.TRAIN)
val_dataset = tfds.load(dataset_name, split = tfds.Split.VALIDATION)
print(train_dataset)
print(val_dataset)

cp_path = 'best_weights.hdf5'
cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path, save_best_only = True, save_weights_only = True, verbose = 2)

pre_trained_model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = (300,300,3))
pre_trained_model.trainable = False

def preprocessing(features):
    image = tf.image.resize(features['image'], size=(300,300))
    print('Final image shape',image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, lower= 0, upper=5)
        image = tf.image.random_brightness(image, 0.2)
    image = tf.divide(image, 255.0)
    label = features['label']
    print('labels shape :',label)
    label = tf.one_hot(features['label'], 102)
    return image, label

def train():
    train_data = train_dataset.map(preprocessing).batch(32)
    val_data = val_dataset.map(preprocessing).batch(32)

    model = tf.keras.Sequential(
        [
            pre_trained_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation = 'relu'),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(1024, activation = 'relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(102, activation = 'softmax')
        ]
    )
    model.compile(optimizer = tf.optimizers.Adam(lr=5.42e-6), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    try:
        model.load_weights(cp_path)
        print('Weights loaded')
    except:
        print('No Previous Weights Found')

    history = model.fit(train_data, epochs = 10, verbose = 1, validation_data = val_data, callbacks = [cp_callback])
    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    return model

if __name__ == '__main__':
    model = train()
    model.save('model.h5')
