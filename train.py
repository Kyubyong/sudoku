# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/sudoku
'''
from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import load_data, get_batch_data
from modules import conv
from tqdm import tqdm

class Graph(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # inputs
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data() # (N, 9, 9)
            else:
                self.x = tf.placeholder(tf.float32, (None, 9, 9))
                self.y = tf.placeholder(tf.int32, (None, 9, 9))
            self.enc = tf.expand_dims(self.x, axis=-1) # (N, 9, 9, 1)
            self.istarget = tf.to_float(tf.equal(self.x, tf.zeros_like(self.x))) # 0: blanks
            
            # network
            for i in range(hp.num_blocks):
                with tf.variable_scope("conv2d_{}".format(i)):
                    self.enc = conv(self.enc, 
                                    filters=hp.num_filters, 
                                    size=hp.filter_size,
                                    is_training=is_training,
                                    norm_type="bn",
                                    activation_fn=tf.nn.relu)
            
            # outputs        
            self.logits = conv(self.enc, 10, 1, scope="logits") # (N, 9, 9, 1)
            self.probs = tf.reduce_max(tf.nn.softmax(self.logits), axis=-1) #( N, 9, 9)
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1)) #( N, 9, 9)
            
            # accuracy
            self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
            self.acc = tf.reduce_sum(self.hits) / (tf.reduce_sum(self.istarget) + 1e-8)
            tf.summary.scalar("acc", self.acc)
                                       
            if is_training:
                # Loss
                self.ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
                self.loss = tf.reduce_sum(self.ce * self.istarget) / (tf.reduce_sum(self.istarget))
                
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                tf.summary.scalar("loss", self.loss)
            
            self.merged = tf.summary.merge_all()
            
def main():
    g = Graph(); print("Training Graph loaded")
    with g.graph.as_default():# Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=60)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)
                    if step%10==0:
                        print(sess.run([g.loss, g.acc]))

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step) 
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
                
if __name__ == "__main__":
    main(); print("Done")
