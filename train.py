# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

# set log level to debug
tf.sg_verbosity(10)

class Hyperparams:
    batch_size = 64

def load_data(is_train=True):
    '''Loads training / validation data.
    
    Args
      is_train: Boolean. If True, it loads training data.
        Otherwise, it loads validation data.
    
    Returns:
      X: 4-D array of float. Has the shape of (# total games, 9, 9, 1) (for train) 
        or (batch_size, 9, 9, 1) (for validation)
      Y: 3-D array of int. Has the shape of (# total games, 9, 9) (for train) 
        or (batch_size, 9, 9) (for validation)            
    '''
    X = np.load('data/sudoku_quizzes.npy').astype(np.float32)
    Y = np.load('data/sudoku_solutions.npy')
    
    X = np.expand_dims(X, -1)
    
    if is_train:
        return X[:-Hyperparams.batch_size], Y[:-Hyperparams.batch_size] # training data
    else:
        return X[-Hyperparams.batch_size:], Y[-Hyperparams.batch_size:] # validation data
        
def get_batch_data(is_train=True):
    '''Returns batch data.
    
    Args:
      is_train: Boolean. If True, it returns batch training data. 
        Otherwise, batch validation data. 
        
    Returns:
      A Tuple of x, y, and num_batch
        x: A `Tensor` of float. Has the shape of (batch_size, 9, 9, 1).
        y: A `Tensor` of int. Has the shape of (batch_size, 9, 9).
        num_batch = A Python int. Number of batches.
    '''
    X, Y = load_data(is_train=is_train)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X), 
                                                  tf.convert_to_tensor(Y)]) 
    
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=Hyperparams.batch_size, 
                                  capacity=Hyperparams.batch_size*64,
                                  min_after_dequeue=Hyperparams.batch_size*32, 
                                  allow_smaller_final_batch=False)
    # calc total batch count
    num_batch = len(X) // batch_size 
    
    return x, y, num_batch  # (64, 9, 9, 1), (64, 9, 9), ()

class Graph(object):
    def __init__(self, is_train=True):
        # inputs
        if is_train:
            self.x, self.y, self.num_batch = get_batch_data()
            self.x_val, self.y_val, _ = get_batch_data(is_train=False)
        else:
            self.x = tf.placeholder(tf.float32, [None, 9, 9, 1])

        with tf.sg_context(size=3, act='relu', bn=True):
            self.logits = self.x.sg_identity()
            for _ in range(10):
                self.logits = (self.logits.sg_conv(dim=512))

            self.logits = self.logits.sg_conv(dim=10, size=1, act='linear', bn=False)
            
        if is_train:
            self.ce = self.logits.sg_ce(target=self.y, mask=False)
            self.istarget = tf.equal(self.x.sg_squeeze(), tf.zeros_like(self.x.sg_squeeze())).sg_float()
            self.loss = self.ce * self.istarget
            self.reduced_loss = self.loss.sg_sum() / self.istarget.sg_sum()
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
            
            # accuracy evaluation ( for validation set )
            self.preds_ = (self.logits.sg_reuse(input=self.x_val).sg_argmax()).sg_int()
            self.hits_ = tf.equal(self.preds_, self.y_val).sg_float()
            self.istarget_ = tf.equal(self.x_val.sg_squeeze(), tf.zeros_like(self.x_val.sg_squeeze())).sg_float()
            self.acc = (self.hits_ * self.istarget_).sg_sum() / self.istarget_.sg_sum()

def main():
    g = Graph()
    
    tf.sg_train(lr=0.0001, lr_reset=True, log_interval=10, save_interval=300, 
                loss=g.reduced_loss, eval_metric=[g.acc], 
                ep_size=g.num_batch, save_dir='asset/train', max_ep=10, early_stop=False)
    
if __name__ == "__main__":
    main(); print "Done"
