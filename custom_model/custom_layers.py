from __future__ import print_function
from custom_model.math_utils import *
import random 
import numpy as np

import tensorflow as tf
from keras import activations, initializers, constraints, regularizers
import keras.backend as K
from keras.layers import Layer, Reshape, Conv2D, BatchNormalization, TimeDistributed, Lambda, Activation, Concatenate
from keras import Model
from keras.layers import Input

dim = 201
Ad, Au, A = directed_adj()
scaled_laplacian = normalized_laplacian(A)

class TemporalAttention(Layer):
    
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
    def compute_output_shape(self, input_shapes):
        return input_shapes

    def build(self, input_shapes):
        T = input_shapes[-3]
        N = input_shapes[-2]
        F = input_shapes[-1]
        
        self.W1 = self.add_weight(shape=(N, 1),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=None,
                                  constraint=None)
        
        self.W2 = self.add_weight(shape=(F, N),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=None,
                                  constraint=None)
        self.W3 = self.add_weight(shape=(F, 1),
                                  initializer=self.kernel_initializer,
                                  name='W3',
                                  regularizer=None,
                                  constraint=None)
        self.Ve = self.add_weight(shape=(T, T),
                                  initializer=self.kernel_initializer,
                                  name='Ve',
                                  regularizer=None,
                                  constraint=None)
        self.be = self.add_weight(shape=(T, T),
                                  initializer=self.bias_initializer,
                                  name='be',
                                  regularizer=None,
                                  constraint=None)
        
        self.built = True

    def call(self, inputs, mask=None):
        x = K.permute_dimensions(inputs, (0,1,3,2)) #(b,T,F,N)
        r1 = K.dot(x,self.W1)
        lhs = K.dot(r1[...,0], self.W2) #(b,T,N)
        r2 = K.dot(inputs,self.W3) #(b,T,N)
        rhs = K.permute_dimensions(r2[...,0], (0,2,1)) #(b,N,T)
        product = K.batch_dot(lhs, rhs) #(b,T,T)
        #E = K.dot(tf.nn.sigmoid(product+self.be), self.Ve)
        E = tf.einsum('jk,ikl->ijl', self.Ve, tf.nn.sigmoid(product+self.be))
        kernel = tf.nn.softmax(E,axis=-2)
        
        conv = tf.einsum('ijkl,ilm->ijkm', K.permute_dimensions(inputs, (0,2,3,1)), kernel)
        
        return K.permute_dimensions(conv, (0,3,1,2))
    

class DynamicGC(Layer):
    def __init__(self,
                 k,
                 units,
                 scaled_lap = scaled_laplacian,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        
        self.k = k
        self.units = units
        self.scaled_lap = scaled_lap

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        
        self.supports = []
        s = K.constant(K.to_dense(calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(s)

        super(DynamicGC, self).__init__(**kwargs)

    def build(self, input_shape):
        F = input_shape[-1]
        N = input_shape[-2]
        T = input_shape[-3]
        
        self.W1 = self.add_weight(shape=(N, N),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=None,
                                  constraint=None)
        
        self.W2 = self.add_weight(shape=(F, N),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=None,
                                  constraint=None)
        self.W3 = self.add_weight(shape=(F, self.units),
                                  initializer=self.kernel_initializer,
                                  name='W3',
                                  regularizer=None,
                                  constraint=None)
        self.bias1 = self.add_weight(shape=(N, ),
                                  initializer=self.kernel_initializer,
                                  name='b1',
                                  regularizer=None,
                                  constraint=None)
        self.bias2 = self.add_weight(shape=(self.units, ),
                                  initializer=self.bias_initializer,
                                  name='b2',
                                  regularizer=None,
                                  constraint=None)


        self.built = True

    def call(self, inputs):
        A = self.supports[0]  # Adjacency matrix (N x N)
            
        feature = K.dot(K.permute_dimensions(inputs, (0,2,1)), A*self.W1)
        dense = K.dot(K.permute_dimensions(feature, (0,2,1)), self.W2)
        dense = K.bias_add(dense, self.bias1)
                
        mask = dense + -10e15 * (1.0 - A)
        mask = K.softmax(mask)

        node_features = tf.einsum('ijk,ikm->ijm', mask, inputs)  # (N x F)
        trans = K.dot(node_features, self.W3)

        out = K.bias_add(trans, self.bias2)


        output = self.activation(out)
        return output

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], self.units)
    
class SpatialAttention(Layer):
    def __init__(self,
                 k,
                 units,
                 scaled_lap = scaled_laplacian,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        
        self.k = k
        self.units = units
        self.scaled_lap = scaled_lap

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        
        self.supports = []
        s = K.constant(K.to_dense(calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(s)

        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        F = input_shape[-1]
        N = input_shape[-2]
        T = input_shape[-3]
        
        self.W1 = self.add_weight(shape=(T, 1),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=None,
                                  constraint=None)
        
        self.W2 = self.add_weight(shape=(F, T),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=None,
                                  constraint=None)
        self.W3 = self.add_weight(shape=(F, 1),
                                  initializer=self.kernel_initializer,
                                  name='W3',
                                  regularizer=None,
                                  constraint=None)
        self.Ve = self.add_weight(shape=(N, N),
                                  initializer=self.kernel_initializer,
                                  name='Ve',
                                  regularizer=None,
                                  constraint=None)
        self.be = self.add_weight(shape=(N, N),
                                  initializer=self.kernel_initializer,
                                  name='be',
                                  regularizer=None,
                                  constraint=None)
        self.kernel = self.add_weight(shape=(F, self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=None,
                                  constraint=None)
        self.bias = self.add_weight(shape=(self.units,),
                                  initializer=self.kernel_initializer,
                                  name='bias',
                                  regularizer=None,
                                  constraint=None)

        self.built = True

    def call(self, inputs):
        A = self.supports[0]  # Adjacency matrix (N x N)
            
        x = K.permute_dimensions(inputs, (0,2,3,1)) #(b,T,F,N)
        r1 = K.dot(x,self.W1)
        lhs = K.dot(r1[...,0], self.W2) #(b,N,T)
        r2 = K.dot(inputs,self.W3) #(b,T,N)
        rhs = r2[...,0]
        product = K.batch_dot(lhs, rhs) #(b,N,N)
        E = tf.einsum('jk,ikm->ijm', self.Ve, tf.nn.sigmoid(product+self.be)) #(b,N,N)
        mask = E + -10e15 * (1.0 - A)
        mask = K.softmax(mask)
        #mask = tf.nn.softplus(mask)
        
        conv = tf.einsum('ijk,ilkm->iljm', mask, inputs)
        p = K.dot(conv, self.kernel)+self.bias
        
        return self.activation(p)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], input_shapes[2], self.units)

class STBlock(Layer):
    def __init__(self,
                 k,
                 units,
                 time_length,
                 scaled_lap = scaled_laplacian,
                 mode = 'dgc',
                 **kwargs):
        
        self.k = k
        self.units = units
        self.time_length = time_length
        self.scaled_lap = scaled_lap
        self.mode = mode

        super(STBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        shapes = (input_shape[-3], input_shape[-2], self.units)
        
        self.ta = TemporalAttention()#, depthwise=False
        self.ta.build(shapes)
        w1 = self.ta.trainable_weights
        
        if self.mode=='dgc':
            self.sa = TimeDistributed(DynamicGC(k=self.k, units=self.units))#, depthwise=False
            self.sa.build(input_shape)
            w2 = self.sa.trainable_weights
        else:
            self.sa = SpatialAttention(k=self.k, units=self.units)#, depthwise=False
            self.sa.build(input_shape)
            w2 = self.sa.trainable_weights
        
        self.tconv = Conv2D(self.units, (self.time_length, 1), padding='same')#, depthwise=False
        self.tconv.build(shapes)
        w3 = self.tconv.trainable_weights
        
        self.rconv = Conv2D(self.units, (self.time_length, 1), padding='same')#, depthwise=False
        self.rconv.build(input_shape)
        w4 = self.rconv.trainable_weights
        
        #self.ln = BatchNormalization(momentum=0.9,epsilon=0.0001)
        #self.ln.build((None, input_shape[-3], input_shape[-2], self.units))
        #w5 = self.ln.trainable_weights self.time_length

        w = w1+w2+w3+w4#+w5
        
        
        self._trainable_weights = w
        self.built = True

    def call(self, inputs):
        x = self.sa(inputs)
        x = self.ta(x)
        x = self.tconv(x)
        
        res = self.rconv(inputs)
        
        output = tf.nn.softplus(res+x)
        
        return output

    def compute_output_shape(self, input_shapes):
        #return (input_shapes[0], input_shapes[1]-self.time_length+1, input_shapes[2], self.units)
        return (input_shapes[0], input_shapes[1], input_shapes[2], self.units)

    
def STAG(obs, pred, k, time_length, nb_units, nb_blocks, mode='dgc'):
    inp = Input(shape=(obs,201,2))
    x = inp
    #x = Lambda(lambda x: K.reverse(x,axes=-3))(inp)
    
    for i in range(nb_blocks):
        x = STBlock(k=k,units=nb_units,time_length=time_length, mode=mode)(x)
        x = BatchNormalization(epsilon=1e-6)(x)

    x = Conv2D(pred, (1, 1),data_format='channels_first', activation=None)(x)
    x = Conv2D(2, (1, 1), activation=None)(x)
    w, k = Lambda(lambda x: tf.split(x,2,axis=-1))(x)
    w = Activation('sigmoid')(w)
    k = Lambda(lambda x: x-1)(k)
    k = Activation('exponential')(k)
    #k = Lambda(lambda x: x+1e-4)(k)
    out = Concatenate(-1)([w,k])
    return Model(inp, out)

def STAG_home(obs, pred, k, time_length, nb_units, nb_blocks, mode='dgc'):
    inp = Input(shape=(obs,201,2))
    x = inp
    #x = Lambda(lambda x: K.reverse(x,axes=-3))(inp)
    
    for i in range(nb_blocks):
        x = STBlock(k=k,units=nb_units,time_length=time_length, mode=mode)(x)
        x = BatchNormalization(epsilon=1e-6)(x)

    x = Conv2D(60, (15, 1), activation='sigmoid')(x)
    out = Reshape((201,60))(x)
    
    return Model(inp, out)

    
def STAG_norm(obs, pred, k, time_length, nb_units, nb_blocks, mode='dgc'):
    inp = Input(shape=(obs,201,2))
    x = inp
    #x = Lambda(lambda x: K.reverse(x,axes=-3))(inp)
    
    for i in range(nb_blocks):
        x = STBlock(k=k,units=nb_units,time_length=time_length, mode=mode)(x)
        x = BatchNormalization(epsilon=1e-6)(x)

    x = Conv2D(pred, (1, 1),data_format='channels_first', activation=None)(x)
    x = Conv2D(2, (1, 1), activation=None)(x)
    w, k = Lambda(lambda x: tf.split(x,2,axis=-1))(x)
    w = Activation('sigmoid')(w)
    #k = Lambda(lambda x: x-1)(k)
    k = Activation('sigmoid')(k)
    k = Lambda(lambda x: x*0.29+1e-4)(k)
    out = Concatenate(-1)([w,k])
    return Model(inp, out)