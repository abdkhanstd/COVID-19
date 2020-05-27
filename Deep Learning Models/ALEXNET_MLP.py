import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
    
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allocator_type = 'BFC'
set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

import argparse
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import keras.metrics
from keras.optimizers import Adam, RMSprop
import numpy as np


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2




import numpy, scipy.io



img_height=224
img_width=224
batch_size=16
# create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size,subset='training')
val_it = datagen.flow_from_directory('data/train/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size,subset='validation')

# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



#Declare model

#####################################################
l2_reg=0.
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()


metrics = ['accuracy']
optimizer = Adam(lr=1e-5, decay=1e-6)
steps_per_epoch = (28526) // batch_size

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=2)
    
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=metrics)
#model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=metrics)

# Train the model
model.fit_generator(train_it, 
epochs=5,
#steps_per_epoch=steps_per_epoch,
validation_data=test_it,
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)],
verbose=1, 
workers=1,
validation_steps=40)



# evaluate model
#loss = model.evaluate_generator(test_it, steps=24)

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_it, 5480 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
tn, fp, fn, tp =confusion_matrix(test_it.classes, y_pred).ravel()
print(tp)
print(tn)
print(fp)
print(fn)
specificity = tn / (tn+fp)
print(specificity)
print('Classification Report')
target_names = ['Negative', 'Positive']
print(classification_report(test_it.classes, y_pred, target_names=target_names))


# ROC
#from sklearn.metrics import roc_curve
#y_pred_keras = model.predict(test_it)
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_it, y_pred_keras)

#from sklearn.metrics import auc
#auc_keras = auc(fpr_keras, tpr_keras)

#print(fpr_keras)
#print(tpr_keras)

#scipy.io.savemat('roc_vgg16_mlp.mat', mdict={'fpr': fpr_keras,'tpr': tpr_keras})


# ROC Curves
#Y_pred = model.predict_generator(test_it, 5480 // batch_size).ravel()
#fpr_rf, tpr_rf, thresholds_rf = roc_curve(test_it.classes, y_pred)
#auc_rf = auc(fpr_rf, tpr_rf)

#scipy.io.savemat('roc_vgg16_mlp.mat', mdict={'fpr': fpr_rf,'tpr': tpr_rf})



