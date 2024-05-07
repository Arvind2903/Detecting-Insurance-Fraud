# Detecting-Insurance-Fraud

## Introduction
Given images of damaged cars, both AI generated and real accidents, build a classifier that can identify an actual accident from a fraudulent one.

## Setup and Dependencies
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- seaborn

## Data
The data can be found here: https://drive.google.com/drive/folders/17I3-H0PTz2Vh3liqIUOXOVjBmD9KJTXG?usp=sharing

## Data Preprocessing
- ImageDataGenerator is used for data augmentation and preprocessing.
- Images are resized to 224x224 pixels.
- Training and validation data are split with a validation split of 15%.

## Model Architecture
- The VGG16 model is loaded with pretrained ImageNet weights.
- The top layers of the VGG16 model are removed, and additional Dense layers are added.
- GlobalAveragePooling2D is used to flatten the output of the base model.
- LeakyReLU activation, Dropout, and BatchNormalization are applied to Dense layers.
- The output layer consists of a single neuron with a sigmoid activation function.

## Training
- Adam optimizer with a learning rate of 0.001 is used.
- Binary crossentropy loss is used as the loss function.
- Metrics tracked during training include accuracy, precision, and recall.
- ModelCheckpoint and EarlyStopping callbacks are used for saving the best weights and early stopping.

## Models
The fitted models can be found here: https://drive.google.com/drive/folders/1LpoJszTF6ya_4MmiIm3EJCDN1Tnl_PjL?usp=sharing

## Evaluation
- Model performance is evaluated on a separate test set.
- Accuracy, Precision, and Recall metrics are calculated.
- Classification report and confusion matrix are generated to evaluate model performance.

## Results
- The model achieves an accuracy of XX%, precision of XX%, and recall of XX% on the test set.

## Conclusion
The VGG16-based image classification model demonstrates effective performance in distinguishing between Fraud and Non-Fraud classes.

