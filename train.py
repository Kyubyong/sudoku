# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

# set log level to debug
tf.sg_verbosity(10)

def load_data(is_train=True, num_mult=10):
    Y = np.load('data/sudoku.npy') # solutions
    
    Y = np.tile(Y, [num_mult, 1, 1]) # augmented *10
    X = np.zeros_like(Y, dtype=np.float32)
    for i, y in enumerate(Y): # game-wise
        masks = np.random.randint(0, 2, (9, 9)) # 0 or 1.
        x = y * masks # puzzle. 0: blanks=targets.
        X[i] = x
    
    X = np.expand_dims(X, -1)
    
    if is_train:
        return X[:-100], Y[:-100] # training data
    else:
        return X[-100:], Y[-100:] # validation data
    
def get_batch_data(is_train=True, batch_size=16):
    '''
    Args:
      is_train: Boolean. If True, load training data. Otherwise, load validation data. 
    Returns:
      A Tuple of X batch queues (Tensor), Y batch queues (Tensor), and number of batches (int) 
    '''
    # Load data
    X, Y = load_data(is_train=is_train)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X), 
                                                  tf.convert_to_tensor(Y)]) 
    
    # create batch queues
    X_batch, Y_batch = tf.train.shuffle_batch(input_queues,
                                      num_threads=8,
                                      batch_size=batch_size, 
                                      capacity=batch_size*64,
                                      min_after_dequeue=batch_size*32, 
                                      allow_smaller_final_batch=False)
    # calc total batch count
    num_batch = len(X) // batch_size 
    
    return X_batch, Y_batch, num_batch  # (16, 9, 9, 1) int32. cf. Y_batch: (16, 9, 9) int32

class Graph(object):
    def __init__(self, is_train=True):
        # inputs
        if is_train:
            self.X, self.Y, self.num_batch = get_batch_data() # (16, 9, 9, 1), (16, 9, 9)
            self.X_val, self.Y_val, _ = get_batch_data(is_train=False)
        else:
            self.X = tf.placeholder(tf.float32, [None, 9, 9, 1])

        with tf.sg_context(size=3, act='relu', bn=True):
            self.logits = self.X.sg_identity()
            for _ in range(5):
                self.logits = (self.logits.sg_conv(dim=512))
            self.logits = self.logits.sg_conv(dim=10, size=1, act='linear', bn=False) # (16, 9, 9, 10) float32
            
        if is_train:
            self.ce = self.logits.sg_ce(target=self.Y, mask=False) # (16, 9, 9) dtype=float32
            self.istarget = tf.equal(self.X.sg_squeeze(), tf.zeros_like(self.X.sg_squeeze())).sg_float() # zeros: 1, non-zeros: 0 (16, 9, 9) dtype=float32
            self.loss = self.ce * self.istarget # (16, 9, 9) dtype=float32
            self.reduced_loss = self.loss.sg_sum() / self.istarget.sg_sum()
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
            
            # accuracy evaluation ( for train set )
            self.preds = (self.logits.sg_argmax()).sg_int()
            self.hits = tf.equal(self.preds, self.Y).sg_float()
            self.acc_train = (self.hits * self.istarget).sg_sum() / self.istarget.sg_sum()
            
            # accuracy evaluation ( for validation set )
            self.preds_ = (self.logits.sg_reuse(input=self.X_val).sg_argmax()).sg_int()
            self.hits_ = tf.equal(self.preds_, self.Y_val).sg_float()
            self.istarget_ = tf.equal(self.X_val.sg_squeeze(), tf.zeros_like(self.X_val.sg_squeeze())).sg_float()
            self.acc_val = (self.hits_ * self.istarget_).sg_sum() / self.istarget_.sg_sum()

def main():
    g = Graph()
    
    tf.sg_train(log_interval=10, lr_reset=True, 
                loss=g.reduced_loss, eval_metric=[g.acc_train, g.acc_val], ep_size=g.num_batch, 
                save_dir='asset/train', max_ep=100, early_stop=False)
    
if __name__ == "__main__":
    main(); print "Done"
