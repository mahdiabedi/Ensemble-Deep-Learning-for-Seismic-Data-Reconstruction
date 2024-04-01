#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENSEMBLE DEEP LEARNING FOR ENHANCED SEISMIC DATA RECONSTRUCTION
Mar 2024
Author: Mohammad Mahdi Abedi

This script contains the implementation of a deep learning model for reconstructing seismic data with consecutive missing traces.
Refer to the article for more information.
"""

# Import necessary libraries
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2DTranspose
from utilities import FK_weight, FKfilter, CustomFKfilterLayer, conv_block, deconv_block, build_unet
#%% Loading Data
# Define the directory to save results
savefolder = 'ResultsTemp'

# Define the number of epochs
epochs = 521  

# Load seismic data with missing traces
X_train_withmissing = np.load('X_train_withmissing.npy')


sh = tf.shape(X_train_withmissing).numpy()
sc = sh[2]  # Number of columns
st = sh[1]  # Number of rows 
ss = sh[0]  # Number of data samples

# Generate positive and negative F-K weights for dip filtering
weight_positive = FK_weight(a=0.1, b=8, sc=sc,st=st, filter_type='dip')
weight_negative = FK_weight(a=0.1, b=-8,sc=sc,st=st, filter_type='dip')

#%% Defining Models

# Clear previous session
tf.keras.backend.clear_session()

# Ensemble model initialization
initializer = tf.random_normal_initializer(0., 0.02, seed=0)

# Input layer
Xinput = Input(shape=[st, sc, 1])

# F-K dip filtering
X = CustomFKfilterLayer(weight=weight_positive)(Xinput)
XblockA_0 = build_unet([32, 64, 128, 128], X)
XblockA = CustomFKfilterLayer(weight=weight_positive, compute_inverse=True)(XblockA_0)

X = CustomFKfilterLayer(weight=weight_negative)(Xinput)
XblockB_0 = build_unet([32, 64, 128, 128], X)
XblockB = CustomFKfilterLayer(weight=weight_negative, compute_inverse=True)(XblockB_0)

# Concatenate outputs of the two baselines with input
X = Concatenate()([XblockA, XblockB, Xinput])

# Final convolutions
X = conv_block(X, 64, kernel_size=3, strides=2, initializer=initializer, use_bias=True)
X = conv_block(X, 32, kernel_size=3, strides=1, initializer=initializer, use_bias=True)
X = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh')(X)

# Create ensemble model
ensemble_model = Model(inputs=Xinput, outputs=[X, XblockA_0, XblockB_0])

# Display model summary
ensemble_model.summary()


#%% Defining Loss Function and Training Step

# Define loss function for the ensemble model
@tf.function
def ensemble_model_loss_stack(model_output, target_all, v1, w):
    # Unpack model outputs
    complete_model_output, blocka_output, blockb_output = model_output
    target, target_a, target_b = target_all
    wa, wb = w
    
    # Compute relative L1 loss at the output of the complete model
    Norm_target_masked = tf.reduce_mean(tf.math.abs(tf.math.multiply(v1, target)))
    diffrence0 = tf.abs(target - complete_model_output)
    Norm_loss_masked = tf.math.divide(tf.reduce_mean(tf.math.multiply(v1, diffrence0)), Norm_target_masked)
    
    # Compute relative L1 loss for block A
    diffrencea = tf.abs(target_a - blocka_output)
    Norm_target_a_masked = tf.reduce_mean(tf.math.abs(tf.math.multiply(v1, target_a)))
    Norm_loss_masked_a = tf.math.divide(tf.reduce_mean(tf.math.multiply(v1, diffrencea)), Norm_target_a_masked)
    
    # Compute simple L1 loss for block B
    diffrenceb = tf.abs(target_b - blockb_output)
    Norm_target_b_masked = tf.reduce_mean(tf.math.abs(tf.math.multiply(v1, target_b)))
    Norm_loss_masked_b = tf.math.divide(tf.reduce_mean(tf.math.multiply(v1, diffrenceb)), Norm_target_b_masked)
    
    # Compute total loss with weighted contributions from block A and block B
    total_loss = Norm_loss_masked + wa * Norm_loss_masked_a + wb * Norm_loss_masked_b
    
    # Aggregate losses for monitoring
    losses = total_loss, Norm_loss_masked_a, Norm_loss_masked_b
    return losses

# Define initial and final learning rates for learning rate scheduler
initial_learning_rate = 0.001
final_learning_rate = 0.0001

# Learning rate scheduler
def lr_schedule(epoch):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate

# Adam optimizer with initial learning rate
optimizer1 = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Compile the ensemble model with the defined optimizer and loss function
ensemble_model.compile(optimizer=optimizer1, loss=ensemble_model_loss_stack)

# Training step function
@tf.function
def train_step(input_image, target, v1, w):
    with tf.GradientTape() as gen_tape:
        model_output = ensemble_model(input_image)
        losses = ensemble_model_loss_stack(model_output, target, v1, w)
        total_loss = losses[0]

    ensemble_model_gradients = gen_tape.gradient(total_loss, ensemble_model.trainable_variables)
    optimizer1.apply_gradients(zip(ensemble_model_gradients, ensemble_model.trainable_variables))
    return losses

#%% Training Loop

def fit(train_dataset, epochs, Augment={}, numofmasks=20):
    # Lists to store training history
    train_history_blockA = []
    train_history_blockB = []
    train_history_total = []

    for epoch in range(0, epochs):
        w_loss = 0.5, 0.5  # The weights of each branch in loss

        # Update learning rate for the optimizer at the beginning of each epoch
        new_learning_rate = lr_schedule(epoch)
        optimizer1.learning_rate.assign(new_learning_rate)

        print("\nEpoch %d / %d, Num masks %s, %s, Loss weight %s, Learning rate: %.5f" % (
            epoch + 1, epochs, numofmasks, savefolder, w_loss, new_learning_rate))
        start_time = time.time()

        # Lists to store losses for each batch
        loss_a = []
        loss_b = []
        loss_total = []

        for steps, (target_data1) in enumerate(train_dataset):

            miss_v = tf.math.minimum(1, tf.math.reduce_max(target_data1 * 100000, axis=1, keepdims=True))

            if 'flip' in Augment:
                flipped_second_half = tf.image.flip_left_right(target_data1[BATCH_SIZE // 2:])
                target_data1 = tf.concat([target_data1[:BATCH_SIZE // 2], flipped_second_half], axis=0)

            if 'polarity' in Augment:
                random_signs = tf.cast(
                    tf.random.uniform(shape=[tf.shape(target_data1)[0]], minval=0, maxval=2, dtype=tf.int32) * 2 - 1,
                    dtype=tf.float32)
                random_signs = tf.expand_dims(tf.expand_dims(tf.expand_dims(random_signs, axis=-1), axis=-1), axis=-1)
                target_data1 = tf.math.multiply(target_data1, random_signs)

            target_a = FKfilter(target_data1, weight=weight_positive)
            target_b = FKfilter(target_data1, weight=weight_negative)

            for step in range(15):  # num Mask realizations for each batch
                nbatch = np.shape(target_data1)[0]
                v1 = np.zeros((nbatch, sc, 1))
                for i in range(0, nbatch):
                    index_center = np.random.randint(low=1, high=sc, dtype=int)
                    indices = range(max(0, index_center - numofmasks // 2),
                                    min(sc, index_center + numofmasks // 2))
                    v1[i, indices] = 1

                v1 = tf.constant(v1, dtype=tf.float32)
                v1 = tf.expand_dims(v1, axis=1)
                v1 = tf.math.multiply(v1, miss_v)

                v0 = abs(1 - v1)

                v1loss = v1

                input_data = tf.math.multiply(v0, target_data1)

                target_all = target_data1, target_a, target_b
                losses0 = train_step(input_data, target_all, v1loss, w_loss)
                loss_total0, loss_a0, loss_b0 = losses0
                loss_a.append(loss_a0)
                loss_b.append(loss_b0)

                loss_total.append(loss_total0)
            print('.', end='', flush=True)
        # Compute epoch losses
        block_a_loss = tf.reduce_mean(loss_a)
        block_b_loss = tf.reduce_mean(loss_b)
        complete_loss = tf.reduce_mean(loss_total)

        print("BlockA: %.6f, BlockB: %.6f,  Total loss: %.6f,  Time taken: %.1fs" %
              (float(block_a_loss), float(block_b_loss), float(complete_loss), time.time() - start_time), end='\n')

        # Append losses to the history lists
        train_history_blockA.append(float(block_a_loss))
        train_history_blockB.append(float(block_b_loss))
        train_history_total.append(float(complete_loss))

        # Save the model and history periodically
        if (epoch) % 40 == 0 or epoch == epochs:
            print('Saving...')
            ensemble_model.save('/ensemble_modelNmask' + 'epoch' + str(epoch))
            history = {
                "block_a_loss": train_history_blockA,
                "block_b_loss": train_history_blockB,
                "total_loss": train_history_total,
                "Parameters": {
                    'weights_in_loss': w_loss,
                    'Augmentation': Augment,
                    'numofmasks': numofmasks}}
            np.save('/my_history.npy', history)

    return history


# %%
print(sh)
BUFFER_SIZE = sh[0] // 2
BATCH_SIZE = 8
train_dataset = tf.data.Dataset.from_tensor_slices(X_train_withmissing)
train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=0, reshuffle_each_iteration=True).batch(BATCH_SIZE)

# Start training
history = fit(train_dataset, epochs=epochs, Augment={'flip', 'polarity'})


