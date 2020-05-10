#!/usr/bin/env python

# -*- encoding: utf-8

'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

@author: Yekun Chai
@license: CASIA
@email: chaiyekun@gmail.com
@file: layers.py
@time: 5/6/20 9:40 PM
@desc：       
               
'''
import tensorflow as tf
import tensorflow_addons as tfa


class Attention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.f1 = tf.keras.layers.Dense(units)
        self.f2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        h_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.f1(features) + self.f2(h_time_axis))
        attn_weights = tf.nn.softmax(self.V(score), 1)
        ctx_vec = tf.reduce_sum(attn_weights * features, 1)
        return ctx_vec, attn_weights


class CNNEncoder(tf.keras.Model):
    def __init__(self, embed_dim):
        super(CNNEncoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNNDecoder(tf.keras.Model):
    def __init__(self, embed_dim, units, vocab_size):
        super().__init__()
        self.units = units

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.rnn = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attn = Attention(units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        ctx_vec, attn_weights = self.attn(features, hidden)
        x = self.embed(x)
        x = tf.concat([tf.expand_dims(ctx_vec, 1), x], -1)
        output, state = self.rnn(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attn_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
