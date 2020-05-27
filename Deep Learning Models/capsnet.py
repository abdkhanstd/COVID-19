from __future__ import print_function
import sys



import tensorflow as tf


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
config.allow_soft_placement=True
config.log_device_placement=False
config.gpu_options.allocator_type = 'BFC'
set_session(tf.Session(config=config))

from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop

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
import scipy.io


# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
img_height=299
img_width=299


# A common Conv2D model
model_ =keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet')
model_.summary()
#Add a layer where input is the output of the  second last layer 
x = Reshape((-1, 2048))(model_.layers[-3].output)
capsule = Capsule(2,16,5, True)(x)
x = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(input=model_.input, output=x)
model.summary()






# we use a margin loss
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model.summary()

# we can compare the performance with or without data augmentation
data_augmentation = True


# create a data generator
datagen = ImageDataGenerator()
batch_size=32
# load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size,subset='training')

# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/',target_size=(img_height, img_width),class_mode='binary', batch_size=batch_size)

metrics = ['accuracy']
optimizer = Adam(lr=1e-5, decay=1e-6)
steps_per_epoch = (28526) // batch_size



# Train the model
history=model.fit_generator(train_it, 
epochs=1,
#steps_per_epoch=steps_per_epoch,
validation_data=test_it,
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)],
verbose=1, 
workers=4,
validation_steps=40)


model.save('capsnet.h5')


# evaluate model
#loss = model.evaluate_generator(test_it, steps=24)

#Confution Matrix and Classification Report
y_pred = model.predict_generator(test_it, 5480 // batch_size+1)

#scipy.io.savemat('history.mat', mdict={'Acc': history.history['acc'],'val_acc': history.history['val_acc']})
#scipy.io.savemat('history_2.mat', mdict={'loss': history.history['loss'],'val_loss': history.history['val_loss']})
#scipy.io.savemat('caps_roc.mat', mdict={'Y_pred': Y_pred,'test_classes': test_it.classes})
#scipy.io.savemat('caps_roc_2.mat', mdict={'Y_pred': Y_pred,'test_classes': test_it.classes})

y_pred=np.argmax(y_pred,axis=1)
labels = (test_it.class_indices)
labels = dict((v,k) for k,v in labels.items())
y_pred = [labels[k] for k in y_pred]

print(y_pred)
print(test_it.classes)
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
