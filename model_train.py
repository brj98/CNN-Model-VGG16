import os                               
import time                              
import numpy as np   
import tensorflow as tf 
from PIL import Image
import numpy as np
from skimage import transform
from keras import backend as K           
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.models import load_model
from keras.preprocessing import image  
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping


dataSet = 'UsingDataBase/'
trainDataSet = dataSet + 'train/'
testDataSet = dataSet + 'test/'

trainNo = trainDataSet + 'normal/'
trainDr = trainDataSet + 'diabetic_retinopathy/'
testNo = testDataSet + 'normal/'
testDr = testDataSet + 'diabetic_retinopathy/'


#counter of files for train folders and test folders
trainFilesList = [trainNo  , trainDr]
testFilesList = [testNo , testDr]

#count and print details ....
trainCount = 0
testCount = 0
print('start counting ....')

for i in trainFilesList:
  print('train counting ...\n')
  trainCount += len(os.listdir(i))

for i in testFilesList:
  print('test counting ....\n')
  testCount += len(os.listdir(i))

print('train count: ' , trainCount , '\n')
print('test count: ' , testCount , '\n')



#select batch size and IMG_SHAPE for VGG16 
IMG_SHAPE = 224
batch_size = 50

img_gen_train = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)
train_data_gen = img_gen_train.flow_from_directory(batch_size = batch_size,
                                                     directory = trainDataSet,
                                                     shuffle= True,
                                                     target_size = (IMG_SHAPE,IMG_SHAPE),
                                                    class_mode  = "binary")
print(train_data_gen.class_indices)








img_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = img_gen_test.flow_from_directory(batch_size=50,
                                                 directory=testDataSet,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')

valid_data_gen = test_data_gen
vgg_application_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

#Stop trainable becuase its from keras applictions and automaticly have a default error values...
for layer in vgg_application_model.layers:
    print(layer.name)
    layer.trainable = False
    
print(len(vgg_application_model.layers))
last_layer = vgg_application_model.get_layer('block5_pool')
#print size of output layer....
#print('last layer output shape:', last_layer.output_shape)
#but it's not necessary
last_output = last_layer.output

x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(vgg_application_model.input, x)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=None,
                        decay=0.0,
                        amsgrad=False)

for layer in model.layers[:15]:
    layer.trainable = False



model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
model.summary()
#Check Point And Early Stopping
# ModelCheckpoint callback - save best weights
CHECKPOINT_PATH = "./checkpoint/"

checkpoint= ModelCheckpoint(filepath=CHECKPOINT_PATH+"eyesDetectionCheckpoint.h5",
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

epochs = 30
history_model = model.fit(train_data_gen,
                        epochs = epochs,
                        validation_data = valid_data_gen,
                        verbose = 1,
                        steps_per_epoch=28,
                        validation_steps=11,
                        callbacks=[checkpoint , early_stop])
model_json = model.to_json()
with open("./checkpoint/eyesDetectionJson.json", "w") as json_file:
    json_file.write(model_json)
# here for saving model history in dectinory library...
history_file = './checkpoint/eyesDetectionHistory.npy'
np.save(history_file ,history_model.history)