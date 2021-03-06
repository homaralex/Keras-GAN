from __future__ import print_function, division

import argparse

import h5py
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Concatenate, Conv2DTranspose
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

from matplotlib import gridspec

from wgan.wgan import wasserstein_loss

SEGMENT_SIZE = 100 * 1000


class WALI():
    def __init__(self):
        self.last_epoch = 0
        self.random_samples = None
        self.name = 'wali'

        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01

        def optimizer():
            return Adam(0.0001, beta_1=0.5, beta_2=0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=[wasserstein_loss],
                                   optimizer=optimizer(),
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=[wasserstein_loss],
                               optimizer=optimizer())

        # Build and compile the encoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss=[wasserstein_loss],
                             optimizer=optimizer())

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
        self.bigan_generator.compile(loss=[wasserstein_loss, wasserstein_loss],
                                     optimizer=optimizer())

    def build_encoder(self):
        img = Input(shape=self.img_shape)
        x = Conv2D(filters=128, kernel_size=5, strides=2)(img)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=256, kernel_size=5, strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=256, kernel_size=5, strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=512, kernel_size=5, strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        z = Dense(self.latent_dim)(x)
        z = Reshape((self.latent_dim,))(z)

        model = Model(img, z)
        model.summary()

        return model

    def build_generator(self):
        noise_shape = (self.latent_dim,)

        model = Sequential()

        some_val = 4
        model.add(Dense(1024 * some_val * some_val, activation="relu", input_shape=noise_shape))
        model.add(Reshape((some_val, some_val, 1024)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", activation='tanh'))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        x_inputs = Input(shape=self.img_shape)

        x = Conv2D(filters=128, kernel_size=5, strides=2)(x_inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=256, kernel_size=5, strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=256, kernel_size=5, strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=512, kernel_size=3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        z_inputs = Input(shape=(self.latent_dim,))
        z = Reshape((1, 1, self.latent_dim))(z_inputs)
        z = Conv2D(filters=1024, kernel_size=1)(z)
        z = Dropout(0.2)(z)
        z = Conv2D(filters=1024, kernel_size=1)(z)
        z = Dropout(0.2)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = Dropout(0.2)(xz)
        xz = Dense(2048)(xz)
        xz = LeakyReLU(0.1)(xz)

        xz = Dropout(0.2)(xz)
        xz = Dense(1)(xz)
        xz = Activation('linear')(xz)

        return Model([z_inputs, x_inputs], xz)

    def train(self, dataset_path, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        h5_file = h5py.File(dataset_path)
        dset = h5_file['images']

        half_batch = int(batch_size / 2)
        d_losses, g_losses, losses_ratio = [], [], []
        for epoch in range(self.last_epoch + 1, epochs):
            for segment_id in range(dset.shape[0] // SEGMENT_SIZE):
                print('Loading segment {}'.format(segment_id))
                X_train = dset[segment_id * SEGMENT_SIZE: (segment_id + 1) * SEGMENT_SIZE]
                for iter_id in range(SEGMENT_SIZE // half_batch):
                    for critic_iter in range(self.n_critic):
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

                        positive_y = np.ones((half_batch, 1))
                        negative_y = -np.ones((half_batch, 1))

                        # Train the discriminator (img -> z is valid, z -> img is fake)
                        d_loss_real = self.discriminator.train_on_batch([z_, imgs], positive_y)
                        d_loss_fake = self.discriminator.train_on_batch([z, imgs_], negative_y)
                        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                        # Clip discriminator weights
                        for l in self.discriminator.layers:
                            weights = l.get_weights()
                            weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                            l.set_weights(weights)

                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    # Sample gaussian noise
                    z = np.random.normal(size=(batch_size, self.latent_dim))
                    # Select a random half batch of images
                    idx = np.random.randint(0, X_train.shape[0], batch_size)
                    imgs = X_train[idx]

                    positive_y = np.ones((batch_size, 1))
                    negative_y = -np.ones((batch_size, 1))
                    # Train the generator (z -> img is valid and img -> z is is invalid)
                    g_loss = self.bigan_generator.train_on_batch([z, imgs], [positive_y, negative_y])

                    # Clip encoder weights
                    for l in self.encoder.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                    # Plot the progress
                    print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
                    if iter_id % 15 == 0:
                        g_losses.append(g_loss[0])
                        d_losses.append(d_loss[0])
                        losses_ratio.append(g_loss[0] / d_loss[0])
                        self.save_losses_hist(g_losses, d_losses, losses_ratio)

                    # If at save interval => save generated image samples
                    if iter_id % save_interval == 0:
                        print('Saving weights and images')
                        self.save_imgs('{}_{}_{}'.format(epoch, segment_id, iter_id))
            self.save_weights(epoch)

    def save_imgs(self, epoch):
        if self.random_samples is None:
            self.random_samples = np.random.normal(size=(100, self.latent_dim))

        gen_imgs = self.generator.predict(self.random_samples)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(100):
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(gen_imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

            fig.savefig("{}/images/bbc_{}.png".format(self.name, epoch), dpi=200)
        plt.close(fig)

    def save_losses_hist(self, g_losses, d_losses, losses_ratio):
        out_dir = self.name
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
        self.generator.save_weights('{}/generator.hdf5'.format(self.name))
        self.encoder.save_weights('{}/encoder.hdf5'.format(self.name))
        self.discriminator.save_weights('{}/discriminator.hdf5'.format(self.name))

        with h5py.File("{}/generator.hdf5".format(self.name), "r+") as f:
            f.attrs.create('last_epoch', epoch)

    def load_weights(self):
        self.generator.load_weights('{}/generator.hdf5'.format(self.name))
        self.encoder.load_weights('{}/encoder.hdf5'.format(self.name))
        self.discriminator.load_weights('{}/discriminator.hdf5'.format(self.name))

        with h5py.File("{}/generator.hdf5".format(self.name), "r") as f:
            self.last_epoch = f.attrs.get('last_epoch', 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WALI')
    parser.add_argument('--dataset_path', type=str, default='/home/alex/datasets/bbc_full_r_pr.hdf5')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000000000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    wali = WALI()
    if args.resume:
        wali.load_weights()
    wali.train(dataset_path=args.dataset_path, epochs=args.epochs, batch_size=args.batch_size, save_interval=200)
