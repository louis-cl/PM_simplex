#!/usr/bin/env python3
# simplex.py 2015/11/16 Louis Clergue <louisclergue@gmail.com> - Kyezil
# Programming assignment for the course of "Mathematical Programming" of BS in Mathematics at FME/UPC (Barcelona)
import sys
import numpy as np;

# That function generates chunks (arrays of lines) corresponding to matrices
def generateChunks(fname, limit):
    with open(fname) as f:
        for _ in range(4): next(f) # skip header 3 lines
        chunk = []
        i = 0
        for line in f:
            if '=' in line:
                i = i+1
                if i > limit : return
                yield chunk
                chunk = []
                continue
            if line != '\n': chunk.append(line)
        yield chunk

class Simplex:
    # Solves PL problem:
    #   min c'x
    #   s.t | Ax = b
    #       | x >= 0
    def __init__(self, filename, verbose=False):
        # Get cost vector (c), constrictions coefs (A), independent term (b)
        self.c, self.A, self.b = [np.loadtxt(A, dtype=np.int) for A in generateChunks(filename, 3)]
        self.verbose = verbose
        self.log("SIMPLEX: Have read c,A,b from %s" % filename)

    def display(self):
        print("VECTOR C:")
        print(self.c)
        print("MATRIX A:")
        print(self.A)
        print("VECTOR B:")
        print(self.b)
    def log(self, msg):
        if self.verbose: print(msg)

def main(argv):
    if len(argv) < 2:
        print("Usage: ./simplex.py data1 data2 ... dataN\nNeeds at least 1 data file")
        sys.exit(1)
    # We have at least one data
    for i in range(1,len(argv)) :
        print("------ Data input %d ------" % i)
        S = Simplex(argv[i], verbose=True)

main(sys.argv)
