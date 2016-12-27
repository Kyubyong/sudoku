#!/usr/bin/python2
"""
This is adapted from https://www.ocf.berkeley.edu/~arel/sudoku/main.html.
Generates 1 million Sudoku games. 
Kyubyong Park. kbpark.linguist@gmail.com www.github.com/kyubyong
"""

import random, copy
import numpy as np

sample  = [ [3,4,1,2,9,7,6,8,5],
            [2,5,6,8,3,4,9,7,1],
            [9,8,7,1,5,6,3,2,4],
            [1,9,2,6,7,5,8,4,3],
            [8,7,5,4,2,3,1,9,6],
            [6,3,4,9,1,8,2,5,7],
            [5,6,3,7,8,9,4,1,2],
            [4,1,9,5,6,2,7,3,8],
            [7,2,8,3,4,1,5,6,9] ]
            
"""
Randomly arrange numbers in a grid while making all rows, columns and
squares (sub-grids) contain the numbers 1 through 9.

For example, "sample" (above) could be the output of this function. """
def construct_puzzle_solution():
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle  = [[0]*9 for i in range(9)] # start with blank puzzle
            rows    = [set(range(1,10)) for i in range(9)] # set of available
            columns = [set(range(1,10)) for i in range(9)] #   numbers for each
            squares = [set(range(1,10)) for i in range(9)] #   row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(squares[(i/3)*3 + j/3])
                    choice  = random.choice(list(choices))
        
                    puzzle[i][j] = choice
        
                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i/3)*3 + j/3].discard(choice)

            # success! every cell is filled.
            return puzzle
            
        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass

"""
Randomly pluck out cells (numbers) from the solved puzzle grid, ensuring that any
plucked number can still be deduced from the remaining cells.

For deduction to be possible, each other cell in the plucked number's row, column,
or square must not be able to contain that number. """
def pluck(puzzle, n=0):

    """
    Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
    in cell "c"? """
    def canBeA(puz, i, j, c):
        v = puz[c/9][c%9]
        if puz[i][j] == v: return True
        if puz[i][j] in range(1,10): return False
            
        for m in range(9): # test row, col, square
            # if not the cell itself, and the mth cell of the group contains the value v, then "no"
            if not (m==c/9 and j==c%9) and puz[m][j] == v: return False
            if not (i==c/9 and m==c%9) and puz[i][m] == v: return False
            if not ((i/3)*3 + m/3==c/9 and (j/3)*3 + m%3==c%9) and puz[(i/3)*3 + m/3][(j/3)*3 + m%3] == v:
                return False

        return True


    """
    starts with a set of all 81 cells, and tries to remove one (randomly) at a time
    but not before checking that the cell can still be deduced from the remaining cells. """
    cells     = set(range(81))
    cellsleft = cells.copy()
    while len(cells) > n and len(cellsleft):
        cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
        cellsleft.discard(cell) # record that we are trying this cell

        # row, col and square record whether another cell in those groups could also take
        # on the value we are trying to pluck. (If another cell can, then we can't use the
        # group to deduce this value.) If all three groups are True, then we cannot pluck
        # this cell and must try another one.
        row = col = square = False

        for i in range(9):
            if i != cell/9:
                if canBeA(puzzle, i, cell%9, cell): row = True
            if i != cell%9:
                if canBeA(puzzle, cell/9, i, cell): col = True
            if not (((cell/9)/3)*3 + i/3 == cell/9 and ((cell/9)%3)*3 + i%3 == cell%9):
                if canBeA(puzzle, ((cell/9)/3)*3 + i/3, ((cell/9)%3)*3 + i%3, cell): square = True

        if row and col and square:
            continue # could not pluck this cell, try again.
        else:
            # this is a pluckable cell!
            puzzle[cell/9][cell%9] = 0 # 0 denotes a blank cell
            cells.discard(cell) # remove from the set of visible cells (pluck it)
            # we don't need to reset "cellsleft" because if a cell was not pluckable
            # earlier, then it will still not be pluckable now (with less information
            # on the board).

    # This is the puzzle we found, in all its glory.
    return (puzzle, len(cells))
    
    
"""
That's it.

If we want to make a puzzle we can do this:
    pluck(construct_puzzle_solution())
    
The following functions are convenience functions for doing just that...
"""



"""
This uses the above functions to create a new puzzle. It attempts to
create one with 28 (by default) given cells, but if it can't, it returns
one with as few givens as it is able to find.

This function actually tries making 100 puzzles (by default) and returns
all of them. The "best" function that follows this one selects the best
one of those.
"""
def run(n = 28, iter=100):
    all_results = {}
#     print "Constructing a sudoku puzzle."
#     print "* creating the solution..."
    a_puzzle_solution = construct_puzzle_solution()
    
#     print "* constructing a puzzle..."
    for i in range(iter):
        puzzle = copy.deepcopy(a_puzzle_solution)
        (result, number_of_cells) = pluck(puzzle, n)
        all_results.setdefault(number_of_cells, []).append(result)
        if number_of_cells <= n: break
 
    return all_results, a_puzzle_solution

def best(set_of_puzzles):
    # Could run some evaluation function here. For now just pick
    # the one with the fewest "givens".
    return set_of_puzzles[min(set_of_puzzles.keys())][0]

def display(puzzle):
    for row in puzzle:
        print ' '.join([str(n or '_') for n in row])

    
# """ Controls starts here """
# results = run(n=0)       # find puzzles with as few givens as possible.
# puzzle  = best(results)  # use the best one of those puzzles.
# display(puzzle)          # display that puzzle.


def main(num):
    '''
    Generates `num` games of Sudoku.
    '''
    quizzes = np.zeros((num, 9, 9), np.int32)
    solutions = np.zeros((num, 9, 9), np.int32)
    for i in range(num):
        all_results, solution = run(n=23, iter=10)
        quiz = best(all_results)
        
        quizzes[i] = quiz
        solutions[i] = solution

        if (i+1) % 1000 == 0:
            print i+1
            np.save('data/sudoku.npz', quizzes=quizzes, solutions=solutions)

if __name__ == "__main__":
    main(1000000)
    print "Done!"