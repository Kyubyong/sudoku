# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from train import Graph

# Adoped from http://elmo.sbs.arizona.edu/sandiway/sudoku/examples.html
# Two easiest, one intermediate, two difficult, one not fun in the order. 
problems = '''\
000260701 
680070090
190004500
820100040
004602900
050003028
009300074
040050036
703018000

100489006
730000040
000001295
007120600
500703008
006095700
914600000
020000037
800512004

020608000
580009700
000040000
370000500
600000004
008000013
000020000
009800036
000306090

000600400
700003600
000091080
000000000
050180003
000306045
040200060
903000000
020000100

200300000
804062003
013800200
000020390
507000621
032006000
020009140
601250809
000001002

020000000
000600003
074080000
000003002
080040010
600500000
000010780
500009000
000000040'''

solutions ='''\
435269781
682571493
197834562
826195347
374682915
951743628
519326874
248957136
763418259

152489376
739256841
468371295
387124659
591763428
246895713
914637582
625948137
873512964

123678945
584239761
967145328
372461589
691583274
458792613
836924157
219857436
745316892

581672439
792843651
364591782
438957216
256184973
179326845
845219367
913768524
627435198

276314958
854962713
913875264
468127395
597328621
132596487
325789146
641253879
789641532

126437958
895621473
374985126
457193862
983246517
612578394
269314785
548769231
731852649'''


def data_process():
    # Convert problem and solution sets to the proper format
    global problems, solutions
    
    nproblems = len(problems.strip().split("\n\n"))
    X = np.zeros((nproblems, 9, 9), np.float32)  
    Y = np.zeros((nproblems, 9, 9), np.float32)  
    
    for i, prob in enumerate(problems.strip().split('\n\n')):
        for j, row in enumerate(prob.splitlines()):
            for k, num in enumerate(row.strip()):
                X[i, j, k] = num

    for i, sol in enumerate(solutions.strip().split('\n\n')):
        for j, row in enumerate(sol.splitlines()):
            for k, num in enumerate(row.strip()):
                Y[i, j, k] = num
                            
    X = np.expand_dims(X, -1)
    
    return X, Y
    
def test1():
    '''
    Predicts all at once.
    '''
    X, Y = data_process()
    g = Graph(is_train=False)
        
    with tf.Session() as sess:
        tf.sg_init(sess)
     
        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
        
        total_blanks, total_hits = 0, 0 
        for x_3d, y_2d in zip(X, Y): # problem-wise x: (9, 9, 1), y: (9, 9)
            x_2d = np.squeeze(x_3d, -1) #(9, 9)
            x_4d = np.expand_dims(x_3d, 0) # (1, 9, 9, 1)
            while 1:   
                logits = sess.run(g.logits, {g.X: x_4d}) # (1, 9, 9, 10) float32
                preds = np.squeeze(np.argmax(logits, axis=-1), 0) # (9, 9) # most probable numbers
                
                expected = y_2d[x_2d == 0]
                got = preds[x_2d == 0]
                hits = np.equal(expected, got).sum()
                
                result = np.where(x_2d == 0, preds, y_2d).astype(int)
                
#                 print result
                print "Acc.=%d/%d=%.2f\n" % (hits, len(expected), float(hits)/len(expected))
                     
                total_blanks += len(expected)
                total_hits += hits
                break
                 
        print "Total Accuracy = %d/%d=%.2f" % (total_hits, total_blanks, float(total_hits)/total_blanks)

def test2():
    '''
    Predicts sequentially.
    '''
    X, Y = data_process()
    g = Graph(is_train=False)
         
    with tf.Session() as sess:
        tf.sg_init(sess)
      
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
         
        total_blanks, total_hits = 0, 0 
        for x_3d, y_2d in zip(X, Y): # problem-wise x: (9, 9, 1), y: (9, 9)
            x_2d = np.squeeze(x_3d, -1) #(9, 9)
            x_4d = np.expand_dims(x_3d, 0) # (1, 9, 9, 1)
            _x_2d = np.copy(x_2d) # (9, 9) 
            while 1:   
                logits = sess.run(g.logits, {g.X: x_4d}) # (1, 9, 9, 10) float32

                def softmax(x):
                    """Compute softmax values for each sets of scores in x."""
                    e_x = np.exp(x - np.max(x, -1, keepdims=True))
                    return e_x / e_x.sum(axis=-1, keepdims=True) 
                
                activated = softmax(logits) # (1, 9, 9, 10) float32
                preds = np.squeeze(np.argmax(activated, axis=-1), 0) # (9, 9) # most probable numbers
                preds_prob = np.squeeze(np.max(activated, axis=-1), 0) # (9, 9) # highest probabilities for blanks
                preds_prob = np.where(x_2d == 0, preds_prob, 0) # (9, 9)

                top1 = np.argmax(preds_prob) # the index of the most confident number amongst all predictions
                ind = np.unravel_index(top1, (9,9)) 
                got = preds[ind] # the most confident number
                x_2d[ind] = got # result
                
                x_4d = np.expand_dims(np.expand_dims(x_2d, 0), -1)
 
                if len(x_2d[x_2d == 0]) == 0:
                    expected = y_2d[_x_2d == 0]
                    got = x_2d[_x_2d == 0]
                    hits = np.equal(expected, got).sum()
                    
                    result = np.where(_x_2d == 0, x_2d, y_2d).astype(int)
                    
                    print result
                    print "Acc.=%d/%d=%.2f\n" % (hits, len(expected), float(hits)/len(expected))
                     
                    total_blanks += len(expected)
                    total_hits += hits
                    break
                 
        print "Total Accuracy = %d/%d=%.2f" % (total_hits, total_blanks, float(total_hits)/total_blanks)
                    
if __name__ == '__main__':
    test1()
#     test2()
    print "Done"
