import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
    
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.visible_device_list = "0"
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allocator_type = 'BFC'
set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

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


#import matplotlib.pyplot as plt


import numpy, scipy.io



img_height=224
img_width=224

# create a data generator
datagen = ImageDataGenerator()
batch_size=16
# load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size,subset='training')
val_it = datagen.flow_from_directory('data/train/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size,subset='validation')


# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



#Declare model
model_ =keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True)





#Add a layer where input is the output of the  second last layer 
x = Dense(2, activation='softmax', name='predictions')(model_.layers[-2].output)
model = Model(input=model_.input, output=x)
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

