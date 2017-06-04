# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
from data_load import *
from modules import *
from tqdm import tqdm

class Graph(object):
    def __init__(self, is_training=True):
        self.is_training = tf.convert_to_tensor(is_training, tf.bool)
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # inputs
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()
            else: # inference
                self.x = tf.placeholder(tf.float32, [None, 9, 9, 1])
            
            self.x_3d = tf.expand_dims(self.x, axis=-1)
            for i in range(hp.num_blocks):
                self.x_ = tf.layers.conv2d(self.x_3d, 
                                     hp.num_filters, 
                                     hp.filter_size, 
                                     padding="same",
                                     name="block_{}".format(i))
                self.x_3d += normalize(self.x_, is_training=self.is_training, activation_fn=tf.nn.relu,
                                    scope="normalize_{}".format(i))
                
            self.logits = tf.layers.conv2d(self.x_3d, 10, 1, name="logits")
    
            if is_training:
                self.ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
                self.istarget = tf.to_float(tf.equal(self.x, tf.zeros_like(self.x)))
                self.loss = tf.reduce_sum(self.ce * self.istarget) / (tf.reduce_sum(self.istarget) + 1e-5)
                
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                tf.summary.scalar("loss", self.loss)
                
                # accuracy evaluation ( for validation set )
                self.preds = tf.to_int32(tf.arg_max(self.logits, -1))
                self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
                self.acc = tf.reduce_sum(self.hits) / (tf.reduce_sum(self.istarget) + 1e-5)
                tf.summary.scalar("acc", self.acc)
                
                self.merged = tf.summary.merge_all()
            
def main():
    # Load eval data
    X, Y = load_data("val")
    
    g = Graph(); print("Training Graph loaded")
    with g.graph.as_default():# Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)
                
                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step) 
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                
                # eval
                acc = sess.run(g.acc, {g.x: X, g.y: Y, g.is_training: False})
                print(acc)
    
if __name__ == "__main__":
    main(); print("Done")
