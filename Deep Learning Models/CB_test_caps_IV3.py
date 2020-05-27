"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""


"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.visible_device_list = "1"
config.allow_soft_placement=True
config.log_device_placement=False

set_session(tf.Session(config=config))

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()



from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from model_caps_iv3 import ResearchModels
from data import DataSet
import functools
import keras.metrics


import numpy as np

from sklearn.metrics import classification_report, confusion_matrix



def validate(data_type, model, seq_length=60, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 16

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=15,
        use_multiprocessing=True,
        workers=1)

    print(results)
    print(rm.model.metrics_names)
    
    
        #val_generator = data.frame_generator(1,'test', data_type)
    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    # Evaluate!
    scores=np.zeros([14])
    total=np.zeros([14])

    val_trues=[]
    val_preds=[]
    for X,y in data.gen_test('test', data_type): 
        results = rm.model.predict(X)
        predicted=np.argmax(results, axis=-1)
        idx=np.where(np.array(y)==1)
        true_label=idx[1]
        print(true_label)

        total[true_label]=total[true_label]+1
        
        print(len(predicted))
        print(len(true_label))
        if predicted[0]==true_label[0]:
            scores[true_label]=scores[true_label]+1
        
        
        

    
    #val_preds = np.argmax(results, axis=-1)
    
 
    print('Confusion Matrix')
    tn, fp, fn, tp =confusion_matrix(true_label, predicted).ravel()
    print("\n *** Confusion matrix**")
    print(tp)
    print(tn)
    print(fp)
    print(fn)
    print('\n****************')
    
    print(classification_report(true_label, predicted))
    print(scores)
    print('\n****************')
    print(total)

def main():
    model = 'lrcn'  #capsule,lstm,conv_3d,lrcn,c3d
    #saved_model = '/home/abdkhan/covid/data/checkpoints_caps/lstm-features.015-0.658-iv3-caps.hdf5'
    #saved_model = '/home/abdkhan/covid/data/checkpoints_caps/lstm-features.028-0.616-iv3-caps.hdf5'
    #saved_model = '/home/abdkhan/covid/data/checkpoints_caps/conv_3d-images.015-0.523-iv3-caps.hdf5'
    #saved_model = '/home/abdkhan/covid/data/checkpoints_caps/lrcn-images.009-2.551-iv3-caps.hdf5'
    #saved_model = '/home/abdkhan/covid/data/checkpoints_caps/c3d-images.004-0.507-iv3-caps.hdf5'
    
    
    ################### Inverted#####################
    saved_model = '/home/abdkhan/covid/data/checkpoints_caps/capsule-features.004-0.582-iv3-caps.hdf5'
    saved_model = '/home/abdkhan/covid/data/checkpoints_caps/lrcn-images.001-3.222-iv3-caps.hdf5'
    
        # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['capsule','lstm']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=2)

if __name__ == '__main__':
    main()
