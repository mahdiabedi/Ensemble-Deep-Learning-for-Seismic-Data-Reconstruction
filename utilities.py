#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 2024
This file contains utility functions and classes for frequency-wavenumber (F-K) filtering and building the U-Net model.
Refer to the paper ENSEMBLE DEEP LEARNING FOR ENHANCED SEISMIC DATA RECONSTRUCTIO
@author: Mohammad Mahdi Abedi
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Concatenate
import numpy as np

#%% Transformation Modules

# Function to generate frequency-wavenumber (F-K) weights for filtering
def FK_weight(a, b, sc, st, fan_dip=1., filter_type='dip'):
    """
    Generate F-K weights for filtering.
    
    Parameters:
        a (float): Parameter controlling the filter shape.
        b (float): Parameter controlling the filter shape.
        fan_dip (float): Parameter for fan filter type.
        filter_type (str): Type of filter to apply ('dip', 'fan_above', or 'fan_under').
        sc (int): Number of columns (traces) in the seismic data.
    
    Returns:
        weight (tf.Tensor): F-K weights tensor for filtering.
    """
    k1 = tf.linspace(-1., 1., sc)
    if filter_type == 'dip':
        weight = 0.5 * (1 + a + (a - 1) * tf.tanh(b * (k1)))
        weight = tf.expand_dims(tf.expand_dims(tf.expand_dims(weight, axis=-1), axis=0), axis=1)
    if filter_type == 'fan_above' or filter_type == 'fan_under':
        weight = np.zeros([sc, st // 2 + 1])
        for f in range(0, st // 2):
            if f > 30:
                d = (fan_dip * (f - 30)) / sc
            else:
                d = 0
            w = 0.5 * (1 + a + (a - 1) * tf.tanh(b * (k1 + d)))
            weight[:, f] = w
        weight = weight + np.flip(weight, axis=0)
        if filter_type == 'fan_under':
            weight = 1 - weight
        weight = tf.expand_dims(tf.expand_dims(weight, axis=0), axis=0)
    return weight

# Function to apply F-K filtering
@tf.function
def FKfilter(input_tx, weight, compute_inverse=False):
    """
    Apply F-K filtering to input seismic data.
    
    Parameters:
        input_tx (tf.Tensor): Input seismic data tensor.
        weight (tf.Tensor): F-K weights tensor.
        compute_inverse (bool): Flag indicating whether to compute inverse filtering.
    
    Returns:
        output_tx (tf.Tensor): Filtered seismic data tensor.
    """
    input_tx_t_AtTheEnd = tf.transpose(input_tx, perm=[0, 3, 2, 1])
    fk1 = tf.signal.fftshift(tf.signal.rfft2d(input_tx_t_AtTheEnd), axes=2)
    if compute_inverse:
        weight = 1.0 / weight
    fk_filtered = tf.multiply(tf.dtypes.cast(fk1, tf.complex64), tf.dtypes.cast(weight, tf.complex64))
    output_tx_t_AtTheEnd = tf.signal.irfft2d(tf.signal.ifftshift(fk_filtered, axes=2))
    output_tx = tf.transpose(output_tx_t_AtTheEnd, perm=[0, 3, 2, 1])
    return output_tx

# Custom layer for F-K filtering
class CustomFKfilterLayer(tf.keras.layers.Layer):
    def __init__(self, weight, compute_inverse=False, **kwargs):
        super(CustomFKfilterLayer, self).__init__(**kwargs)
        self.weight = weight
        self.compute_inverse = compute_inverse

    def build(self, input_shape):
        # No need to define w in the build method
        input_shape = input_shape
    
    def call(self, input_tx):
        if self.compute_inverse:
            self.weight = 1.0 / self.weight

        input_tx_t_AtTheEnd = tf.transpose(input_tx, perm=[0, 3, 2, 1])
        fk1 = tf.signal.fftshift(tf.signal.rfft2d(input_tx_t_AtTheEnd), axes=2)
        fk_filtered = tf.multiply(fk1, tf.dtypes.cast(self.weight, tf.complex64))

        output_tx_t_AtTheEnd = tf.signal.irfft2d(tf.signal.ifftshift(fk_filtered, axes=2))
        output_tx = tf.transpose(output_tx_t_AtTheEnd, perm=[0, 3, 2, 1])
        return output_tx

    def get_config(self):
        config = super(CustomFKfilterLayer, self).get_config()
        config['weight'] = self.weight.numpy()  # Convert the tensor to a numpy array for serialization
        config['compute_inverse'] = self.compute_inverse
        return config

   
#%% Model Builder Functions

# Clear previous session
tf.keras.backend.clear_session()

def conv_block(X, filters, kernel_size=3, strides=2, use_batch_norm=True, initializer=tf.random_normal_initializer(0., 0.02, seed=0), use_bias=False):
    """
    Convolutional block for building the U-Net model.
    
    Parameters:
        X (tf.Tensor): Input tensor.
        filters (int): Number of filters.
        kernel_size (int): Size of the convolutional kernel.
        strides (int or list): Stride size for the convolution operation.
        use_batch_norm (bool): Whether to use batch normalization.
        initializer (tf.keras.initializers.Initializer): Initializer for kernel weights.
        use_bias (bool): Whether to use bias in convolutional layers.
    
    Returns:
        X (tf.Tensor): Output tensor.
    """
    X = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=use_bias)(X)
    if use_batch_norm:
        X = BatchNormalization()(X)
    X = ReLU()(X)
    return X

def deconv_block(X, skip_connection, filters, kernel_size=3, strides=2, initializer=tf.random_normal_initializer(0., 0.02, seed=0), use_bias=False):
    """
    Deconvolutional block for building the U-Net model.
    
    Parameters:
        X (tf.Tensor): Input tensor.
        skip_connection (tf.Tensor): Skip connection tensor from the encoder path.
        filters (int): Number of filters.
        kernel_size (int): Size of the deconvolutional kernel.
        strides (int or list): Stride size for the deconvolution operation.
        initializer (tf.keras.initializers.Initializer): Initializer for kernel weights.
        use_bias (bool): Whether to use bias in deconvolutional layers.
    
    Returns:
        X (tf.Tensor): Output tensor.
    """
    X = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=use_bias)(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Concatenate()([skip_connection, X])
    return X

def build_unet(num_features, X, initializer=tf.random_normal_initializer(0., 0.02, seed=0)):
    """
    Build the U-Net model.
    
    Parameters:
        num_features (list): List containing the number of features for each layer.
        X (tf.Tensor): Input tensor.
        initializer (tf.keras.initializers.Initializer): Initializer for kernel weights.
    
    Returns:
        X (tf.Tensor): Output tensor.
    """
    # First convolution block without batch normalization
    X = conv_block(X, num_features[0], kernel_size=5, strides=[4, 2], use_batch_norm=False, initializer=initializer)
    down_layers = [X]

    # Downward path
    for filters in num_features[1:]:
        X = conv_block(X, filters, initializer=initializer)
        down_layers.append(X)
    
    # Upward path
    for i, filters in reversed(list(enumerate(num_features[:-1]))):
        X = deconv_block(X, down_layers[i], filters, initializer=initializer)

    # Final convolution
    X = Conv2DTranspose(1, 5, strides=[4, 2], padding='same', kernel_initializer=initializer, use_bias=True)(X)
    return X
