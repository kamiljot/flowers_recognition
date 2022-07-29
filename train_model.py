import tensorflow as tf
import os
import keras
import numpy as np
from tensorflow.keras import layers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9



#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)



train_ds = tf.keras.preprocessing.image_dataset_from_directory('dataset/train',
                                                               labels='inferred',
                                                               label_mode='categorical',
                                                               class_names=None,
                                                               color_mode='rgb',
                                                               batch_size=32,
                                                               image_size=(512, 512),
                                                               shuffle=True,
                                                               seed=123,
                                                               validation_split=0.2,
                                                               subset="training")
val_ds = tf.keras.preprocessing.image_dataset_from_directory('dataset/val',
                                                               labels='inferred',
                                                               label_mode='categorical',
                                                               class_names=None,
                                                               color_mode='rgb',
                                                               batch_size=32,
                                                               image_size=(512, 512),
                                                               shuffle=True,
                                                               seed=123,
                                                               validation_split=0.2,
                                                               subset="validation")

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 104

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)


















#AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_ds = configure_for_perf(train_ds)


#train_images = list(train_ds.map(lambda x, y: x))
#train_labels = list(train_ds.map(lambda x, y: y))

#model = keras.applications.Xception(weights=None, input_shape=(512, 512, 3), classes=104)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit(train_ds, epochs=10, validation_data=val_ds)