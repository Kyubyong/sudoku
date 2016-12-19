# Can Convolutional Neural Networks Crack Sudoku Puzzles?

This project is motivated simply by my personal curiosity--can CNNs crack Sudoku? There are many approaches to computationally solve Sudoku. Why not neural networks?

## Requirements
  * numpy >= 1.11.1
  * sugartensor >= 0.0.1.8 (pip install sugartensor)
	
## Research Question
Can Convolutional Neural Networks Crack Sudoku Puzzles?

## Background
* To see what Sudoku is, check the wikipedia [here](https://en.wikipedia.org/wiki/Sudoku)
* To investigate this task comprehensively, read through [McGuire et al. 2013](https://arxiv.org/pdf/1201.0749.pdf)

## Workflow
STEP 1. Generate [10,000 Sudoku solutions](https://drive.google.com/open?id=0B0ZXk88koS2KVFBYcnU3RWxzaU0). (Actually more data would be helpful.) (=Y)<br/>
STEP 2. Make blanks randomly with fifty-fifty uniform probabilities for every cell. (=X)<br/>
STEP 3. Build convolutional networks that yield the output of the same shape as the input like this.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5 convolutional layers of 512 dimensions<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 final convolutional layer with a 1 by 1 filter.<br/>
STEP 4. Train the model, feeding X and Y. Loss is calucated from the predictions for the blanks.<br/>
STEP 5. Evaluate.

## Results
After 60 epochs (=361,891 batches), we got [this model file](https://drive.google.com/open?id=0B0ZXk88koS2KU0ZYYTVOOWhqdDA). Subsequently, we evaluated according to the following two methods.
(

* Test Approach 1: Predict the numbers in blanks all at once.
* Test Approach 2: Predict the numbers sequentially from the most confident one at each step.

| Level  |  Test1 | Test2  |
| ---    |---     |---     |
|  Easy | 28/45=0.62|42/45=0.93  |
|  Easy | 29/45=0.64|39/45=0.87  |
| Intermediate  | 23/57=0.40|12/57=0.21  |
| Difficult | 26/58=0.45|13/58=0.22|
| Difficult | 21/47=0.45|5/47=0.11|
| Not Fun | 17/62=0.27|14/62=0.23|
|Total |   144/314=0.46|125/314=0.40|
	






