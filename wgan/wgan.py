# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

from __future__ import print_function, division

import argparse
from functools import partial

import h5py
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Deconv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import _Merge
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
import os
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

Z_DIMS = 100
SEGMENT_SIZE = 100 * 1000


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = kwargs['batch_size']

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class WGAN():
    def __init__(self, batch_size, improved=True):
        self.improved = improved
        # gradient penalty factor
        self.lmbd = 10
        self.batch_size = batch_size

        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01

        def optimizer():
            if self.improved:
                return Adam(0.0001, beta_1=0.5, beta_2=0.9)
            else:
                return RMSprop(lr=0.00005)

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=wasserstein_loss, optimizer=optimizer())

        # The generator takes noise as input and generated imgs
        z = Input(shape=(Z_DIMS,))
        gen_img = self.generator(z)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(with_bn=not self.improved)

        if self.improved:
            real_img = Input(shape=(self.img_rows, self.img_cols, self.channels))
            discriminator_out_real = self.discriminator(real_img)
            discriminator_out_fake = self.discriminator(gen_img)

            # We also need to generate weighted-averages of real and generated samples,
            # to use for the gradient norm penalty.
            averaged_samples = RandomWeightedAverage(batch_size=self.batch_size)([real_img, gen_img])
            # We then run these samples through the discriminator as well. Note that we never really use the
            # discriminator output for these samples - we're only running them to get the gradient norm for the
            # gradient penalty loss.
            averaged_samples_out = self.discriminator(averaged_samples)

            self.discriminator_model = Model(inputs=[real_img, z],
                                             outputs=[discriminator_out_real,
                                                      discriminator_out_fake,
                                                      averaged_samples_out])
            # The gradient penalty loss function requires the input averaged samples to get gradients. However,
            # Keras loss functions can only have two arguments, y_true and y_pred. We get around this
            # by making a partial() of the function with the averaged samples here.
            partial_gp_loss = partial(gradient_penalty_loss,
                                      averaged_samples=averaged_samples,
                                      gradient_penalty_weight=self.lmbd)
            # Functions need names or Keras will throw an error
            partial_gp_loss.__name__ = 'gradient_penalty'
            self.discriminator_model.compile(optimizer=optimizer(),
                                             loss=[wasserstein_loss,
                                                   wasserstein_loss,
                                                   partial_gp_loss])

        else:
            self.discriminator.compile(loss=wasserstein_loss,
                                       optimizer=optimizer(),
                                       metrics=['accuracy'])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(gen_img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss=wasserstein_loss,
                              optimizer=optimizer(),
                              metrics=['accuracy'])

        self.last_epoch = 0
        # for saving images
        self.random_samples = None

    def build_generator(self):

        noise_shape = (Z_DIMS,)

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

    def build_discriminator(self, with_bn):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(128, kernel_size=4, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        if with_bn:
            model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        if with_bn:
            model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1024, kernel_size=4, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())

        model.summary()

        img = Input(shape=img_shape)
        features = model(img)
        valid = Dense(1, activation="linear")(features)

        return Model(img, valid)

    def train(self, dataset_path, epochs, save_interval=50):

        # Load the dataset
        h5_file = h5py.File(dataset_path)
        dset = h5_file['images']

        half_batch = int(self.batch_size / 2)

        d_losses, g_losses, losses_ratio = [], [], []
        for epoch in range(self.last_epoch + 1, epochs):
            for segment_id in range(dset.shape[0] // SEGMENT_SIZE):
                print('Loading segment {}'.format(segment_id))
                X_train = dset[segment_id * SEGMENT_SIZE: (segment_id + 1) * SEGMENT_SIZE]
                for iter_id in range(SEGMENT_SIZE // half_batch):
                    for _ in range(self.n_critic):

                        # ---------------------
                        #  Train Discriminator
                        # ---------------------

                        if self.improved:
                            half_batch = self.batch_size

                        # Select a random half batch of images
                        idx = np.random.randint(0, X_train.shape[0], half_batch)
                        imgs = X_train[idx]

                        if self.improved:
                            # Improved WGAN - sample from uniform
                            noise = np.random.uniform(0, 1, (half_batch, Z_DIMS))
                        else:
                            noise = np.random.normal(0, 1, (half_batch, Z_DIMS))

                        # Generate a half batch of new images
                        gen_imgs = self.generator.predict(noise)

                        # Train the discriminator
                        if not self.improved:
                            positive_y = -np.ones((half_batch, 1))
                            d_loss_real = self.discriminator.train_on_batch(imgs, positive_y)
                            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                            # Clip discriminator weights
                            for l in self.discriminator.layers:
                                weights = l.get_weights()
                                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                                l.set_weights(weights)
                        else:
                            positive_y = np.ones((self.batch_size, 1), dtype=np.float32)
                            negative_y = -positive_y
                            dummy_y = np.zeros((self.batch_size, 1), dtype=np.float32)
                            d_loss = self.discriminator_model.train_on_batch([imgs, noise],
                                                                             [positive_y, negative_y, dummy_y])

                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    if self.improved:
                        # Improved WGAN - sample from uniform
                        noise = np.random.uniform(0, 1, (self.batch_size, Z_DIMS))
                    else:
                        noise = np.random.normal(0, 1, (self.batch_size, Z_DIMS))

                    # Train the generator
                    g_loss = self.combined.train_on_batch(noise, positive_y)

                    # Plot the progress
                    if self.improved:
                        g_loss = g_loss[0]
                        d_loss = sum(d_loss)
                        print("%d %d %d [D loss: %f] [G loss: %f]" % (epoch, segment_id, iter_id, d_loss, g_loss))

                    else:
                        d_loss = 1 - d_loss[0]
                        g_loss = 1 - g_loss[0]
                        print("%d %d %d [D loss: %f] [G loss: %f]" % (epoch, segment_id, iter_id, d_loss, g_loss))
                    if iter_id % 5 == 0:
                        g_losses.append(g_loss)
                        d_losses.append(d_loss)
                        losses_ratio.append(g_losses[-1] / d_losses[-1])
                        self.save_losses_hist(g_losses, d_losses, losses_ratio)

                    # If at save interval => save generated image samples
                    if iter_id % save_interval == 0 and iter_id > 0:
                        self.save_imgs('{}_{}_{}'.format(epoch, segment_id, iter_id))
            self.save_weights(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        if self.random_samples is None:
            self.random_samples = np.random.normal(0, 1, (r * c, Z_DIMS))
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
        fig.savefig("wgan/images/bbc_%d.png" % epoch)
        plt.close()

    def save_losses_hist(self, g_losses, d_losses, losses_ratio):
        out_dir = 'wgan'
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
        self.generator.save_weights('wgan/generator.hdf5')
        self.discriminator.save_weights('wgan/discriminator.hdf5')
        self.combined.save_weights('wgan/combined.hdf5')

        with h5py.File("wgan/generator.hdf5", "r+") as f:
            f.attrs.create('last_epoch', epoch)

    def load_weights(self):
        self.generator.load_weights('wgan/generator.hdf5')
        self.discriminator.load_weights('wgan/discriminator.hdf5')
        self.combined.load_weights('wgan/combined.hdf5')

        with h5py.File("wgan/generator.hdf5", "r") as f:
            self.last_epoch = f.attrs.get('last_epoch', 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--dataset_path', type=str, default='/home/alex/datasets/bbc_full_r_pr.hdf5')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000000000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--improved', action='store_true')
    args = parser.parse_args()

    wgan = WGAN(batch_size=args.batch_size, improved=args.improved)
    if args.resume:
        wgan.load_weights()
    wgan.train(dataset_path=args.dataset_path, epochs=args.epochs, save_interval=200)
