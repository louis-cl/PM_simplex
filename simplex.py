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

    # Get cost vector (c), constrictions coefs (A), independent term (b)
    def __init__(self, filename, verbose=False):
        self.c, self.A, self.b = [np.loadtxt(A) for A in generateChunks(filename, 3)]
        self.verbose = verbose
        self.M, self.N = self.A.shape # store # or original variables
        self.solved = False # mark as unsolved
        self.log("SIMPLEX: Have read c,A,b from %s" % filename)

    # return leaving basic variable, or -1 if no negative reduced cost (optimal)
    def _blandRule(self, c, Binv) :
        pass
    # apply simplex algorithm given a basis, its inverse, and a basic feasible solution
    def _simplex(self, x, c, B, Binv):
        pass

    def _phaseI(self):
        self.log("SIMPLEX: Phase I started")
        self.log("Phase I: make b >= 0", 1)
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i,:] *= -1
                self.b[i] *= -1
        self.log("Phase I: add auxiliary variables",1)
        self.A = np.c_[self.A, np.eye(self.M)]
        B = np.arange(self.N, self.N+self.M) # create starting Base with artificial variables
        Binv = np.eye(len(B)) # inverse of identity is indentity
        y = self.b # SBF is B
        c = np.concatenate((np.zeros(self.N), np.ones(self.M)))
        self.log("Phase I: starting primal simplex with Bland's rule",1)
        self._simplex(y,c,B,Binv)

    def solve(self):
        B = self._phaseI()

    def display(self):
        print("VECTOR C:")
        print(self.c)
        print("MATRIX A:")
        print(self.A)
        print("VECTOR B:")
        print(self.b)
    def log(self, msg, indent=0):
        if self.verbose: print('\t'*indent + msg)

def main(argv):
    if len(argv) < 2:
        print("Usage: ./simplex.py data1 data2 ... dataN\nNeeds at least 1 data file")
        sys.exit(1)
    # We have at least one data
    for i in range(1,len(argv)) :
        print("------ Data input %d ------" % i)
        S = Simplex(argv[i], verbose=True)
        S.solve();

main(sys.argv)
