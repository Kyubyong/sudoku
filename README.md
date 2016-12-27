# Can Convolutional Neural Networks Crack Sudoku Puzzles?

Sudoku is a popular number puzzle that requires you to fill blanks in a 9X9 grid with digits so that each column, each row, and each of the nine 3×3 subgrids contains all of the digits from 1 to 9. There have been various approaches to that, including computational ones. In this pilot project, we show that convolutional neural networks have the potential to crack Sukoku puzzles without any other rule-based post-processing.

## Requirements
  * numpy >= 1.11.1
  * sugartensor >= 0.0.1.8 (pip install sugartensor)
	
## Research Question
Can Convolutional Neural Networks Crack Sudoku Puzzles?

## Background
* To see what Sudoku is, check the [wikipedia](https://en.wikipedia.org/wiki/Sudoku)
* To investigate this task comprehensively, read through [McGuire et al. 2013](https://arxiv.org/pdf/1201.0749.pdf).

## Training
* STEP 1. Generate 1 million Sudoku games. (See `generate_sudoku.py`). The pre-generated games are available [here](https://www.kaggle.com/bryanpark/sudoku/downloads/sudoku.zip).
* STEP 2. Construct convolutional networks as follows. (See `Graph` in `train.py`)<br/>
![graph](graph.png?raw=true)
* STEP 3. Train the model, feeding X (quizzes) and Y (solutions). Note that only the predictions for the position of the blanks count when computing loss. (See `train.py`)<br/>

## Evaluation
We test the performance of the final model against 30 real Sudoku puzzles and their solutions, which vary from the easy to evil level. Metrics are the following two.
* Accuracy: the number of blanks where our prediction matched to the solution.
* Success rate: the number of games where 100% accurately matched to our prediction. 

## Results
After 4 epochs, we got [the best model file](https://drive.google.com/open?id=0B0ZXk88koS2KV1VIT2RYUGhuOEU). We designed two test methods.

* Test method 1: Predict the numbers in blanks all at once.
* Test method 2: Predict the numbers sequentially the most confident one at a time.

 
| Level  |  Test1 <br/>(#correct/#blanks=acc.)| Test2 <br/>(#correct/#blanks=acc.) |
| ---    |---     |---     |
|Easy|43/47=0.91|**47/47=1.00**|
|Easy|37/45=0.82|**45/45=1.00**|
|Easy|40/47=0.85|**47/47=1.00**|
|Easy|33/45=0.73|**45/45=1.00**|
|Easy|37/47=0.79|**47/47=1.00**|
|Easy|39/46=0.85|**46/46=1.00**|
|Medium|27/53=0.51|32/53=0.60|
|Medium|27/55=0.49|27/55=0.49|
|Medium|32/55=0.58|36/55=0.65|
|Medium|28/53=0.53|**53/53=1.00**|
|Medium|27/52=0.52|33/52=0.63|
|Medium|29/56=0.52|39/56=0.70|
|Hard|30/56=0.54|41/56=0.73|
|Hard|31/55=0.56|28/55=0.51|
|Hard|33/55=0.60|**55/55=1.00**|
|Hard|33/57=0.58|**57/57=1.00**|
|Hard|27/55=0.49|50/55=0.91|
|Hard|28/56=0.50|27/56=0.48|
|Expert|32/56=0.57|22/56=0.39|
|Expert|32/55=0.58|**55/55=1.00**|
|Expert|37/54=0.69|**54/54=1.00**|
|Expert|33/55=0.60|**55/55=1.00**|
|Expert|30/55=0.55|23/55=0.42|
|Expert|25/54=0.46|**54/54=1.00**|
|Evil|32/50=0.64|**50/50=1.00**|
|Evil|33/50=0.66|**50/50=1.00**|
|Evil|34/49=0.69|**49/49=1.00**|
|Evil|33/53=0.62|**53/53=1.00**|
|Evil|35/51=0.69|**51/51=1.00**|
|Evil|34/51=0.67|**51/51=1.00**|
|Total Accuracy| 971/1568=0.62| **1322/1568=0.84**|
|Success Rate| 0/30=0| **19/30=0.63**|

## Conclusions
* I also tested fully connected layers, to no avail.
* Up to some point, it seems that CNNs can learn to solve Sudoku.
* For most problems, the second method was outperform the fist one.
* Humans cannot predict all numbers simultaneously. Probably so do CNNs.

## Furthery Study
* Reinforcement learning would be more appropriate for Sudoku solving.

## Notes for reproducibility
* Download pre-generated Sudoku games [here](https://www.kaggle.com/bryanpark/sudoku/downloads/sudoku.zip) and extract it to `data/` folder.
* Download the pre-trained model file [here](https://drive.google.com/open?id=0B0ZXk88koS2KV1VIT2RYUGhuOEU) and extract it to `asset/train/ckpt` folder.
	






