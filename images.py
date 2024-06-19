import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.__version__)

EPOCHS = 20
TRAIN_MODEL = False
LOAD_MODEL = not TRAIN_MODEL

batch_size = 32
img_height = 180
img_width = 180

import pathlib



dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
### TODO:CHANGE THIS TO MY NEW PATH! ###
### Reicht wenn ich eine ähnliche Verzeichnisstruktur für meine Fotos nutze
### --> Klassen + Labels werden automatisch gebildet ;-)
archive = '/Users/floriancarstens/.keras/datasets/flower_photos' #tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')
print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

### Create a dataset

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

"""You can find the class names in the `class_names` attribute on these datasets."""

class_names = train_ds.class_names
print(class_names)

if TRAIN_MODEL:
  """### Visualize the data
  
  Here are the first nine images from the training dataset.
  """

  print('show images')
  plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
  plt.show()

  ## augmentation
  data_augmentation = keras.Sequential(
    [
      layers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
      layers.RandomCrop(img_height,img_width),
    ]
  )

  #show augmented images
  print('augmented images 1')
  plt.figure(figsize=(10, 10))
  for images, _ in train_ds.take(1):
    for i in range(9):
      augmented_images = data_augmentation(images)
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_images[0].numpy().astype("uint8"))
      plt.axis("off")
  plt.show()

  print('augmented images 2')
  plt.figure(figsize=(10, 10))
  for images, _ in train_ds.take(2):
    for i in range(9):
      augmented_images = data_augmentation(images)
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_images[0].numpy().astype("uint8"))
      plt.axis("off")
  plt.show()

  """You can train a model using these datasets by passing them to `model.fit` (shown later in this tutorial). If you like, you can also manually iterate over the dataset and retrieve batches of images:"""

  for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

  """The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.
  
  You can call `.numpy()` on either of these tensors to convert them to a `numpy.ndarray`.
  
  ### Standardize the data
  
  The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network; in general you should seek to make your input values small.
  
  Here, you will standardize values to be in the `[0, 1]` range by using `tf.keras.layers.Rescaling`:
  """

  normalization_layer = tf.keras.layers.Rescaling(1./255)

  """There are two ways to use this layer. You can apply it to the dataset by calling `Dataset.map`:"""

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  # Notice the pixel values are now in `[0,1]`.
  print(np.min(first_image), np.max(first_image))



  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  """### Train a model
  
  For completeness, you will show how to train a simple model using the datasets you have just prepared.
  
  The [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) model consists of three convolution blocks (`tf.keras.layers.Conv2D`) with a max pooling layer (`tf.keras.layers.MaxPooling2D`) in each of them. There's a fully-connected layer (`tf.keras.layers.Dense`) with 128 units on top of it that is activated by a ReLU activation function (`'relu'`). This model has not been tuned in any way—the goal is to show you the mechanics using the datasets you just created. To learn more about image classification, visit the [Image classification](../images/classification.ipynb) tutorial.
  """

  print('Training model...')

  num_classes = 5

  model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
  ])

  # compile model
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  """Note: You will only train for a few epochs so this tutorial runs quickly."""
  # train model
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
  )

  # visualize training results
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(EPOCHS)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

  # save model
  #!mkdir -p saved_model
  model.save('saved_model/my_model')

if LOAD_MODEL:
  print("Loading model...")
  model = tf.keras.models.load_model('saved_model/my_model')

  # Check its architecture
  #model.summary()

#############################################################################
# try network out
#############################################################################

# take pic
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = '/Users/floriancarstens/.keras/test_pics/Red_sunflower.jpg'#tf.keras.utils.get_file('Red_sunflower.jpg', origin=sunflower_url, cache_subdir='test_pics')
# create image array
# image path
img_path = '/Users/floriancarstens/.keras/test_pics/'
# image filename
img_filename = 'Dandelion3.png'#'Red_sunflower.jpg'
# image path + name
img_path_filename = os.path.join(img_path, img_filename)
print(img_path_filename)
# load image
img = tf.keras.utils.load_img(
    img_path_filename, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# do predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#############################################################################
# save model or load lite model
#############################################################################
# Convert the Keras Sequential model to a TensorFlow Lite model which can run on Jetson MC
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

# Save the model.
#with open('model.tflite', 'wb') as f:
#  f.write(tflite_model)

# run the light model
#TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
#interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

#interpreter.get_signature_list()

#classify_lite = interpreter.get_signature_runner('serving_default')
#classify_lite

#predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
#score_lite = tf.nn.softmax(predictions_lite)
# say prediction
#print(
#    "This image most likely belongs to {} with a {:.2f} percent confidence."
#    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
#)
# compare with original model
#print(np.max(np.abs(predictions - predictions_lite)))
