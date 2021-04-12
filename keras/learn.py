import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import os.path
import csv
from math import ceil
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

dataset = 'chairs' # one of 'faces' 'chairs' 'celeba'
z_dims = 32
beta = 10


batches = 2000000
batch_size = 64

if dataset == 'faces':
    channels = 1
    likelihood = 'bernoulli'
    data = np.load('faces-labelled.npz')['images']
    epochs = ceil(batches / ceil(len(data) / batch_size))
elif dataset == 'chairs':
    channels = 1
    likelihood = 'bernoulli'
    data = np.load('{}.npy'.format(dataset))
    epochs = ceil(batches / ceil(len(data) / batch_size))
elif dataset == 'celeba':
    channels = 3
    likelihood = 'gaussian'
    dir_iter = DirectoryIterator(
        directory='/tmp/tmp',
        image_data_generator=ImageDataGenerator(
            rescale=1/255,
            #validation_split=
        ),
        target_size=(64, 64),
        color_mode='rgb', # grayscale
        class_mode='input',
        batch_size=batch_size,
        shuffle=True,
        #seed= for shuffling
        #subset='training'
        interpolation='bilinear',
    )
    epochs = ceil(batches / len(dir_iter))

class NormalSampler(keras.layers.Layer):
    def __init__(self):
        super(NormalSampler, self).__init__()

    def call(self, mu_logvar):
        mu, logvar = mu_logvar
        epsilon = K.random_normal(shape=K.shape(mu))
        std = K.exp(logvar / 2)
        return mu + epsilon*std


inputs = keras.Input(shape=(channels, 64, 64))

# encoder
x = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(inputs)
x = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
x = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
x = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=256, activation='relu')(x)
mu = keras.layers.Dense(units=z_dims, activation=None)(x)
logvar = keras.layers.Dense(units=z_dims, activation=None)(x)
z = NormalSampler()([mu, logvar])

encoder = keras.Model(inputs=inputs, outputs=[z, mu, logvar], name='encoder')


# decoder
d_inputs = keras.Input(shape=(z_dims,))
x = keras.layers.Dense(units=256, activation='relu')(d_inputs)
x = keras.layers.Dense(units=64*4*4, activation='relu')(x)
x = keras.layers.Reshape((64, 4, 4))(x)
x = keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
x = keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
x = keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
x = keras.layers.Conv2DTranspose(filters=channels, kernel_size=4, strides=2, padding='same', activation=None)(x)
decoder = keras.Model(inputs=d_inputs, outputs=x, name='decoder')


# combined
outputs = decoder(encoder(inputs)[0])
model = keras.Model(inputs=inputs, outputs=outputs)


optimizer = keras.optimizers.Adam(lr=1e-4)

def loss(x, x_recon):
    var = K.exp(logvar)
    mu_sq = K.square(mu)
        
    d_kl = -0.5 * K.sum(1 + logvar - var - mu_sq, axis=1)
    
    if likelihood == 'bernoulli':
        # sigmoid followed by binary cross entropy loss
        # let phi = 1+e^(-x_recon)
        # sigma(x_recon) = 1/phi
        # bce(p, q) = -p*log(q) - (1-p)*log(1-q)
        # bce(x, sigma(x_recon)) = -x*log(1/phi) - (1-x)*log(1-1/phi)
        #  = x*log(phi) - (1-x)*log((phi-1)/phi)
        #  = x*log(phi) - (1-x)*log(e^(-x_recon)/phi)
        #  = x*log(phi) - (1-x)*(-x_recon - log(phi))
        #  = x*log(phi) + (1-x)*x_recon + (1-x)*log(phi)
        #  = log(phi) + (1-x)*x_recon
        recon_loss = K.log(1 + K.exp(-x_recon)) + (1-x)*x_recon
    elif likelihood == 'gaussian':
        x_recon = K.sigmoid(x_recon)
        #recon_loss = K.square(x_recon) - 2*x*x_recon + K.square(x)
        recon_loss = K.square(x_recon - x)
    recon_loss = K.sum(recon_loss, axis=(1,2,3))

    return K.mean(recon_loss + beta*d_kl) # mean over batch



model.compile(optimizer=optimizer, loss=loss)



initial_epoch = 0

model_path = os.path.join('checkpoints', '{}-{}-{}'.format(dataset, z_dims, beta))
log_path = os.path.join('checkpoints', '{}-{}-{}.csv'.format(dataset, z_dims, beta))
if os.path.exists(model_path):
    model.load_weights(model_path)
    with open(log_path, newline='') as f:
        reader = csv.reader(f)
        initial_epoch = int(list(reader)[-1][0]) + 1
    print('Loaded existing checkpoint')

if dataset == 'celeba':
    model.fit_generator(
        dir_iter,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True),
            keras.callbacks.CSVLogger(log_path, append=True),
        ],
    #    validation_split=0,
    )
elif dataset == 'faces' or dataset == 'chairs':
    model.fit(
        x=data,
        y=data,
        batch_size=batch_size,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True),
            keras.callbacks.CSVLogger(log_path, append=True),
        ],
    #    validation_split=0,
    )
