import numpy as np
import pandas as pd 
import sys
import os
from keras import layers
from keras.layers import Input, Dense, Flatten
from keras import Model
from keras import backend as K
from keras.preprocessing import image
from keras.utils import np_utils
from tensorflow.python.client import device_lib
import json
print(device_lib.list_local_devices())


people_dir = sys.argv[1]
dic = {

}
i=0
for dir in os.listdir(people_dir):
    dic.update({dir: i})
    i +=1
print(dic)

def read_csv_image(name_of_file):
    df = pd.read_csv(name_of_file)
    df = np.asarray(df)
    X_train = df[:, 1:]
    name_of_file = name_of_file.split('/')
    print(name_of_file)
    name = name_of_file[-1]
    name = name.split('_')
    print(dic[name[0]])
    y_train = np.full(X_train.shape[0], dic[name[0]])
    print(dic[name[0]])
    return X_train, y_train


data_dir = sys.argv[2]
X_train = np.empty((1, 512))
y_train = np.empty(1)
for dir in os.listdir(data_dir):
    print(dir)
    print(os.path.join(data_dir, dir))
    x_people, y_people = read_csv_image(os.path.join(data_dir, dir))
    X_train = np.concatenate((X_train, x_people), axis=0)
    y_train = np.concatenate((y_train, y_people), axis=0)
X_train = X_train[1:, :]
y_train = y_train[1:]
y_train = np_utils.to_categorical(y_train, 55)
print(y_train.shape)

#model
def model(input_shape):
    x_input = Input(input_shape)
    X  = Dense(256, activation = 'relu', name = 'layer1')(x_input)
    X  = Dense(128, activation = 'relu', name = 'layer2')(X)
    X  = Dense(55, activation = 'softmax', name = 'layer3')(X)
    model = Model(input= x_input, output = X, name='Classification')
    return model



CDCN = model((512,))
CDCN.compile(optimizer = 'adam', loss= "categorical_crossentropy", metrics=['accuracy'])
CDCN.summary()
CDCN.fit(x = X_train, y = y_train, epochs=20, batch_size=16)
CDCN.save('CDCN.h5')
CDCN_detail = CDCN.to_json()
print(CDCN_detail)
with open('CDCN.json', 'w') as outfile:  
	outfile.write(CDCN_detail)
