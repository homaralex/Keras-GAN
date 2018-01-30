from __future__ import print_function, division

import argparse

import h5py
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import os


class BIGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'],
                               optimizer=optimizer)

        # Build and compile the encoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=['binary_crossentropy'],
                             optimizer=optimizer)

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from samples noise
        z = Input(shape=(self.latent_dim,))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                     optimizer=optimizer)

        self.last_epoch = 0
        self.random_samples = None

    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim,))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)

    def train(self, dataset_path, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        h5_file = h5py.File(dataset_path)
        dset = h5_file['images']
        X_train = dset[:100000]

        half_batch = int(batch_size / 2)

        d_losses, g_losses, losses_ratio = [], [], []
        for epoch in range(self.last_epoch + 1, epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(half_batch, self.latent_dim))
            imgs_ = self.generator.predict(z)

            # Select a random half batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            z_ = self.encoder.predict(imgs)

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample gaussian noise
            z = np.random.normal(size=(batch_size, self.latent_dim))

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            if epoch % 100 == 0:
                g_losses.append(g_loss[0])
                d_losses.append(d_loss[0])
                losses_ratio.append(g_loss[0] / d_loss[0])
                self.save_losses_hist(g_losses, d_losses, losses_ratio)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print('Saving weights and images')
                self.save_imgs(epoch)
                self.save_weights(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        if self.random_samples is None:
            self.random_samples = np.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.generator.predict(self.random_samples)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("bigan/images/bbc_%d.png" % epoch)
        plt.close()

    def save_losses_hist(self, g_losses, d_losses, losses_ratio):
        out_dir = 'bigan'
        plt.plot(g_losses, label='Gen')
        plt.plot(d_losses, label='Dis')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'loss_hist.png'))
        plt.close()

        plt.plot(losses_ratio, label='G / D')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'losses_ratio.png'))
        plt.close()

    def save_weights(self, epoch):
        self.generator.save_weights('bigan/generator.hdf5')
        self.encoder.save_weights('bigan/encoder.hdf5')
        self.discriminator.save_weights('bigan/discriminator.hdf5')

        with h5py.File("bigan/generator.hdf5", "r+") as f:
            f.attrs.create('last_epoch', epoch)

    def load_weights(self):
        self.generator.load_weights('bigan/generator.hdf5')
        self.encoder.load_weights('bigan/encoder.hdf5')
        self.discriminator.load_weights('bigan/discriminator.hdf5')

        with h5py.File("bigan/generator.hdf5", "r") as f:
            self.last_epoch = f.attrs.get('last_epoch', 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiGAN')
    parser.add_argument('--dataset_path', type=str, default='/home/alex/datasets/bbc_full_r_pr.hdf5')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000000000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    bigan = BIGAN()
    if args.resume:
        bigan.load_weights()
    bigan.train(dataset_path=args.dataset_path, epochs=args.epochs, batch_size=args.batch_size, save_interval=200)
