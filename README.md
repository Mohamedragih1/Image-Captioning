
# Image Captioning with RNN and Transfer Learning

This project implements an image captioning pipeline using transfer learning with the VGG16 model for feature extraction and an RNN for sequence generation. Below, you will find a detailed breakdown of the workflow, from data preprocessing to model training and real-time implementation.

## Table of Contents
1. [Setup and Dependencies](#setup-and-dependencies)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Extraction](#feature-extraction)
4. [Text Preprocessing](#text-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Model Training](#model-training)
7. [Real-Time Implementation](#real-time-implementation)
8. [How to Use](#how-to-use)

---

## 1. Setup and Dependencies

The following libraries are required for this notebook:
- `numpy`
- `tensorflow`
- `pickle`
- `os`
- `tqdm`

The project is designed to run in a Kaggle environment with access to the Flickr8k dataset.

```python
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
```

---

## 2. Data Preprocessing

### Directories
Define the base and working directories for the dataset and output files:

```python
BASE_DIR = '/kaggle/input/flickr8k'
WORKING_DIR = '/kaggle/working'
```

### Image Loading
Images are loaded from the dataset folder and resized to 224x224 pixels for compatibility with the VGG16 model.

---

## 3. Feature Extraction

The VGG16 model is used for feature extraction. The last dense layer of the model is removed to use the extracted features as input to the RNN.

```python
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
```

Features are extracted for each image and saved to a pickle file for later use:

```python
features = {}
directory = os.path.join(BASE_DIR, 'Images')

for img_name in tqdm(os.listdir(directory)):
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
```

---

## 4. Text Preprocessing

Tokenize the captions and create sequences for model training. Padding is applied to ensure uniform sequence lengths.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
```

---

## 5. Model Architecture

The model consists of an embedding layer, an LSTM for sequence generation, and a dense layer for output. It combines image features and textual embeddings using a merge layer.

```python
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(fe1)
fe2 = BatchNormalization()(fe2)

inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.3)(se1)
se3 = SimpleRNN(512, return_sequences=False)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(512, activation='relu')(decoder1)
decoder2 = Dropout(0.2)(decoder2)
decoder3 = Dense(256, activation='relu')(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder3)
```

---

## 6. Model Training

The model is trained on the processed features and captions using categorical cross-entropy loss. The training pipeline includes checkpointing for saving the best model.

```python
for i in range(epochs):
    train_generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    
    history = model.fit(
        train_generator, 
        epochs=1, 
        steps_per_epoch=steps, 
        validation_data=val_generator, 
        validation_steps=len(val) // batch_size,  
        verbose=1
    )
```
    results: Train Loss: 2.2998 - Train Accuracy: 0.4500
    Validation Loss: 4.2376 - Validation Accuracy: 0.3069
---

## 7. Real-Time Implementation

The trained model can generate captions for new images. Provide an image path and preprocess it to feed it into the pipeline.

```python
def generate_caption(image_name):

    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
generate_caption("Image path")
```

---



## 9. How to Use

1. Clone this repository or download the notebook.
2. Place the Flickr8k dataset in the specified `BASE_DIR`.
3. Run the notebook cells sequentially to preprocess data, train the model, and generate captions.

