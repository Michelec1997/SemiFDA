import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

BN=16

def my_validation(y_test,y_pred):
    y_test_new=np.zeros(len(y_test))
    y_pred_new=np.zeros(len(y_pred))
    for i in range(len(y_test)):
        y_test_new[i]=np.argmax(y_test[i])
        y_pred_new[i]=np.argmax(y_pred[i])

    accuracy = accuracy_score(y_test_new, y_pred_new)
    #f1 = f1_score(y_test_new, y_pred_new)
    #print(accuracy)
    
    return accuracy

#////////////////////
#AUTOENCODER
#\\\\\\\\\\\\\\\\\\\\
def create_ae(window_size, features):

    input_series = tf.keras.layers.Input(shape=(window_size, features))

    conv_1=tf.keras.layers.Conv1D(filters=4,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',activation='relu')(input_series)

    max_pool_1=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_1)

    ### 1st layer
    conv_2=tf.keras.layers.Conv1D(filters=8,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_1)

    max_pool_2=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_2)


    conv_3=tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_2)

    max_pool_3=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_3)

    conv_4=tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_3)

    max_pool_4=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_4)




    flat_2 = tf.keras.layers.Flatten()(max_pool_4)

    dense_1=tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu)(flat_2)
    dense_2=tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(dense_1)

    reshape_1=tf.keras.layers.Reshape((4,32),input_shape=(128,))(dense_2)


    transpose_conv_1=tf.keras.layers.Conv1DTranspose(filters=32,
                                            kernel_size=5,
                                            strides=2,
                                            padding='same',activation='relu')(reshape_1)



    transpose_conv_2=tf.keras.layers.Conv1DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',activation='relu')(transpose_conv_1)


    transpose_conv_3=tf.keras.layers.Conv1DTranspose(filters=8,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',activation='relu')(transpose_conv_2)

    transpose_conv_4=tf.keras.layers.Conv1DTranspose(filters=4,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',activation='relu')(transpose_conv_3)

    transpose_conv_5=tf.keras.layers.Conv1DTranspose(filters=features,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same',activation='tanh')(transpose_conv_4)


    model = tf.keras.Model(input_series, transpose_conv_5, name='model')

    return model


def create_encoder_cov(window_size, features):

    input_series = tf.keras.layers.Input(shape=(window_size, features))

    conv_1=tf.keras.layers.Conv1D(filters=4,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',activation='relu')(input_series)

    max_pool_1=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_1)

    ### 1st layer
    conv_2=tf.keras.layers.Conv1D(filters=8,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_1)

    max_pool_2=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_2)


    conv_3=tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_2)

    max_pool_3=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_3)

    conv_4=tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_3)

    max_pool_4=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_4)


    flat_2 = tf.keras.layers.Flatten()(max_pool_4)

    dense_1=tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu)(flat_2)

    cov = tf.keras.layers.Lambda(cov_calc, output_shape=(BN,BN ), name='cov')(dense_1)

    model = tf.keras.Model(input_series, cov, name='model')

    return model


def create_full(window_size, features, num_classes):

    input_series = tf.keras.layers.Input(shape=(window_size, features))

    conv_1=tf.keras.layers.Conv1D(filters=4,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',activation='relu')(input_series)

    max_pool_1=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_1)

    ### 1st layer
    conv_2=tf.keras.layers.Conv1D(filters=8,
                                    kernel_size=5,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_1)

    max_pool_2=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_2)


    conv_3=tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_2)

    max_pool_3=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_3)

    conv_4=tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',activation='relu')(max_pool_3)

    max_pool_4=tf.keras.layers.MaxPooling1D(pool_size=2, strides=(2), padding='same')(conv_4)

    flat_2 = tf.keras.layers.Flatten()(max_pool_4)

    dense_1=tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu)(flat_2)

    
    dense_class1=tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu,activity_regularizer=tf.keras.regularizers.L2(0.01))(dense_1)
    drop_class1=tf.keras.layers.Dropout(0.3)(dense_class1)
    dense_class2=tf.keras.layers.Dense(units=8, activation=tf.keras.activations.relu,activity_regularizer=tf.keras.regularizers.L2(0.01))(drop_class1)
    drop_class2=tf.keras.layers.Dropout(0.3)(dense_class2)
    dense_class3=tf.keras.layers.Dense(units=num_classes, activation='softmax')(drop_class2)


    model = tf.keras.Model(input_series, dense_class3, name='model')

    return model



#//////////////
#CLASSIFICATION HEAD
#\\\\\\\\\\\\\\
def create_classification_head(num_classes):

    classifier = tf.keras.Sequential()
    bottleneck_size=16
    classifier.add(tf.keras.layers.InputLayer(input_shape=(bottleneck_size)))

    classifier.add(tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu,activity_regularizer=tf.keras.regularizers.L2(0.01)))
    classifier.add(tf.keras.layers.Dropout(0.3))
    classifier.add(tf.keras.layers.Dense(units=8, activation=tf.keras.activations.relu,activity_regularizer=tf.keras.regularizers.L2(0.01)))
    classifier.add(tf.keras.layers.Dropout(0.3))
    classifier.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    return classifier

def my_loss(y_true,y_pred):
  frobenius_norm = (0.25*tf.square(1/BN))*tf.reduce_sum(tf.square(y_true-y_pred))
  return frobenius_norm 


def cov_calc(args):
  Xs_enc = args

  # Reshape
  batch_size = tf.cast(tf.shape(Xs_enc)[0], Xs_enc.dtype)
  EPS = np.finfo(np.float32).eps

  factor = 1. / (batch_size-1+EPS)

  avg = tf.reduce_mean(Xs_enc, axis=0)

  cov = factor * tf.matmul(tf.transpose(tf.subtract(Xs_enc, avg)), tf.subtract(Xs_enc, avg))

  return cov

def create_client_models(model, n_user, lr):
    #instantiate N-1 clients
    client_models=[tf.keras.Sequential() for _ in range(n_user)]
    for c in range (0,n_user):
        client_models[c]=tf.keras.models.clone_model(model)

    i=0
    for cm in client_models:
        cm.set_weights(model.get_weights())
        cm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy')
        
        i+=1

    return client_models


def create_client_models_ssfl(model, n_user, lr):
    #instantiate N-1 clients
    client_models=[tf.keras.Sequential() for _ in range(n_user)]
    for c in range (0,n_user):
        client_models[c]=tf.keras.models.clone_model(model)

    i=0
    for cm in client_models:
        cm.set_weights(model.get_weights())
        cm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        
        i+=1

    return client_models



def create_client_models_SemiFDA(model, n_user, lr):
    #loss_weights=[1, 1]
    #instantiate N-1 clients
    client_models=[tf.keras.Sequential() for _ in range(n_user)]
    for c in range (0,n_user):
        client_models[c]=tf.keras.models.clone_model(model)

    #opt=keras.optimizers.Adam(learning_rate=lr)

    i=0
    for cm in client_models:
        cm.set_weights(model.get_weights())
        cm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=[my_loss]) 
        i+=1

    return client_models


##########################################################################################################
#ARCHITECTURE for DIGIT CLASSIFICATION (IMAGES)

# def create_full_2d(window_size, features, num_classes):

#     input_series = tf.keras.layers.Input(shape=(window_size, window_size, features))

#     conv_1_0=tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', strides=1, padding='same')(input_series)
#     bn1= tf.keras.layers.BatchNormalization()(conv_1_0)
#     max_pool_1=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn1)

#     conv_2_0=tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', strides=1, padding='same')(max_pool_1)
#     bn2= tf.keras.layers.BatchNormalization()(conv_2_0)
#     max_pool_2=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)

#     conv_3_0=tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', strides=1, padding='same')(max_pool_2)
#     bn3= tf.keras.layers.BatchNormalization()(conv_3_0)
#     max_pool_3=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn3)

#     conv_4_0=tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', strides=1, padding='same')(max_pool_3)
#     bn4= tf.keras.layers.BatchNormalization()(conv_4_0)
#     max_pool_4=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn4)

#     flat = tf.keras.layers.Flatten()(max_pool_4)

#     dense_bn=tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(flat)

#     dense_out=tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)(dense_bn)

#     model = tf.keras.Model(input_series, dense_out, name='model')

#     return model