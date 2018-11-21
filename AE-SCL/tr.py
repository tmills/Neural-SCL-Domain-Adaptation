from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
import pre
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from numpy.random import seed
from keras.models import load_model
import gc
seed(1)




def train(src,dest,pivot_num,pivot_min_st,dim,pivot_method='mi', pivots=None):
    outputs = pivot_num
    HUs =dim
    #get the representation learning training and vlidation data
    x, y, x_valid, y_valid,inputs= pre.preproc(pivot_num,pivot_min_st,src,dest,pivot_method=pivot_method,pivots=pivots)

    model = Sequential()
    model.add(Dense(HUs,kernel_initializer='glorot_normal', input_shape=(inputs,)))
    #model.add(Dense(HUs, input_shape=(inputs,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(outputs))
    model.add(Activation('sigmoid'))
    print(model.summary())
#     opt = SGD(lr=0.1, decay=1e-5, momentum=0.9)
    opt = Adam()

    model.compile(optimizer=opt, loss='binary_crossentropy')

    #stops as soon as the validaion loss stops decreasing
    earlyStopping = EarlyStopping(monitor='val_loss', patience=0, mode='min')
    #saveing only the best model
    save_best = ModelCheckpoint("best_model", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    h=model.fit(x, y, batch_size=50,callbacks=[earlyStopping,save_best],epochs=40,validation_data=(x_valid,y_valid), shuffle=True)
    print( (h.history['val_loss'])[-1])
    weight_str = src + "_to_" + dest + "/weights/w_"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)+"_"+str(HUs)
    filename = weight_str
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    #saving the entire model
    model = load_model("best_model")
    np.save(weight_str, model.get_weights())
    del model
    gc.collect()
