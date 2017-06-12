# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/sudoku
'''

from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp

def normalize(inputs, 
              type="bn",
              decay=.99,
              is_training=True, 
              activation_fn=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but 
        the last dimension. Or if type is `ln`, the normalization is over 
        the last dimension. Note that this is different from the native 
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py` 
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type=="bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        
        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank==2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank==3:
                inputs = tf.expand_dims(inputs, axis=1)
            
            outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                               decay=decay,
                                               center=True, 
                                               scale=True, 
                                               activation_fn=None, 
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               zero_debias_moving_mean=True,
                                               fused=True)
            # restore original shape
            if inputs_rank==2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank==3:
                outputs = tf.squeeze(outputs, axis=1)
        else: # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                               decay=decay,
                                               center=True, 
                                               scale=True, 
                                               activation_fn=activation_fn, 
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               fused=False)    
    elif type=="ln":
        outputs = tf.contrib.layers.layer_norm(inputs=inputs, 
                                            center=True, 
                                            scale=True, 
                                            activation_fn=None, 
                                            scope=scope)
    elif type=="in": # instance normalization
        with tf.variable_scope(scope):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            
            mean, variance = tf.nn.moments(inputs, [1], keep_dims=True)
            gamma = tf.get_variable("gamma", 
                                    shape=params_shape, 
                                    dtype=tf.float32, 
                                    initializer=tf.ones_initializer)
            beta = tf.get_variable("beta", 
                                    shape=params_shape, 
                                    dtype=tf.float32, 
                                    initializer=tf.zeros_initializer)
            normalized = (inputs - mean) / tf.sqrt(variance+1e-8)
            outputs = normalized * gamma + beta
            
    else: # None
        outputs = inputs
    
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    
    return outputs

def conv(inputs, 
         filters=None, 
         size=1, 
         rate=1, 
         padding="SAME", 
         use_bias=False,
         is_training=True,
         activation_fn=None,
         decay=0.99,
         norm_type=None,
         scope="conv",
         reuse=None):
    '''Applies convolution to `inputs`.
    
    Args:
      inputs: A 3D or 4D tensor with shape of [batch, (height), width, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      is_training: A boolean.
      decay: A float of (0, 1).
      activation_fn: A string.
      norm_type: Either `bn`, `ln`, or `in`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A tensor of the same shape and dtypes as `inputs`.
    '''
    ndims = inputs.get_shape().ndims
    conv_fn = tf.layers.conv1d if ndims==3 else tf.layers.conv2d
    
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            assert ndims==3, "if causal is true, the rank must be 3."
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        
        if filters is None:
            filters = inputs.get_shape().as_list[-1]
        
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, 
                "use_bias":use_bias, "reuse":reuse}
        outputs = conv_fn(**params)
        outputs = normalize(outputs, type=norm_type, decay=decay, 
                  is_training=is_training, activation_fn=activation_fn)
    return outputs

