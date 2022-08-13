import matplotlib.pyplot as plt
import sklearn 
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import numpy as np 

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights('model.h5')

nb_train_samples = 28709
nb_validation_samples = 3589
validation_data_dir = 'C:/Python/dataset_AI/PrivateTest'
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
   validation_data_dir,
   color_mode = 'grayscale',
   target_size=(48,48),
   batch_size = 64,
   class_mode = 'categorical',
   shuffle=False)
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

#confution matrix and classification report
Y_pred = emotion_model.predict(validation_generator,nb_validation_samples//64+1 )
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes,y_pred))
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(validation_generator.classes, y_pred,target_names=target_names))

plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(validation_generator.classes,y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_= plt.xticks(tick_marks, classes, rotation=90)

_= plt.yticks(tick_marks,classes)