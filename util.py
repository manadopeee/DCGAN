import tensorflow as tf
import math

def relu(x):
    return tf.maximum(0, x)

def leaky_relu(x, leak=0.02):
    return tf.maximum(x, x * leak)

def Tanh(x):
    tanh = 1 - math.tanh**(x)
    return tanh