import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras import backend as K

def nll_unibeta(mode=2):
    def inloss(y_true, y_pred):
        
        value = y_true
        w = y_pred[...,:1]
        k = y_pred[...,1:]+1e-4
        #k = 1/(y_pred[...,1]+1e-5)-1
        
        a = w*k+1
        b = (1-w)*k+1
        
        n1 = tfp.distributions.Beta(a, b)
        
        loss = n1.prob(value)
        summ = tf.math.log(loss+1e-10) * -1
        
        return summ
    return inloss

def nll_test(mode=2):
    def inloss(y_true, y_pred):
        
        value = y_true
        w = y_pred[...,:1]
        k = 1/(y_pred[...,1:]**2+1e-4)+0.2
        w = w*0.98+0.01
        
        a = w*k+1
        b = (1-w)*k+1
        
        n1 = tfp.distributions.Beta(a, b)
        
        loss = n1.prob(value)
        summ = tf.math.log(loss+1e-10) * -1
        
        return summ
    return inloss


def nll_beta(mode=2):
    def inloss(y_true, y_pred):
        
        value = y_true
        mu, sigma = tf.split(y_pred, 2, axis=-1)
        sigma = sigma + 1e-3
        
#         a = mu**2/sigma**2+2.
#         b = (a-1)*mu
        
        b = sigma**2/(mu**3+1e-4)
        
        #n1 = tfp.distributions.InverseGamma(a, b)
        n1 = tfp.distributions.InverseGaussian(mu, b)
        
        loss = n1.prob(value)
        summ = tf.math.log(loss+1e-7) * -1
        
        return summ
    return inloss

def Gaussian_NLL(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))

    logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def GaussianRegression(y_true, evidential_output):
    mu, logvar = tf.split(evidential_output, 2, axis=-1)
    return Gaussian_NLL(y_true, mu, logvar+1e-4, reduce=True)

def focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed