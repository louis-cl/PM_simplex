#!/usr/bin/env python3
# simplex.py 2015/11/16 Louis Clergue <louisclergue@gmail.com> - Kyezil
# Programming assignment for the course of "Mathematical Programming" of BS in Mathematics at FME/UPC (Barcelona)
import sys # get parameters
import numpy as np; # linear algebra package
from math import inf as INF; # infinity

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

    # return entring basic variable, or -1 if no negative reduced cost (optimal)
    def _blandRule(self, c, B, Binv) :
        l = np.dot(c[B], Binv) # cB*Binv = lambda from dual
        # generator to get reduced costs (computes on the fly)
        r = ( (j, c[j] - np.inner(l,self.A[:,j])) for j in range(self.A.shape[1]) if j not in B)
        # return (rj, j) in increasing order, blandRule -> pick first < 0
        rj = next(r)
        while rj != None and rj[1] >= 0 :
            rj = next(r,None) # return None at end
        return (-1,0) if rj == None else rj # rj == None => rj >= 0 for all j => optimal

    # apply simplex algorithm given a basis, its inverse, and a basic feasible solution
    def _simplex(self, x, c, B, Binv):
        iteration = 1
        z = np.inner(c,x)
        self.log("simplex it 0, starting with z = {}".format(z),2)
        while iteration < 100 : # avoid infinite loop if bug
            q , rq = self._blandRule(c, B, Binv) # get q, entring BV
            if q == -1 :
                self.log("simplex it {:2d}: Optimal solution found".format(iteration),2)
                return True, x, B, np.inner(c,x) # x is optimal with basis B, recompute z
            # compute directions d = -Binv*A[q], we can remove the -
            u = np.dot(Binv, self.A[:,q])
            # select B(p) leaving variable
            theta = INF
            p = -1
            for i in range(len(u)):
                if u[i] > 0 :
                    theta_i = x[B[i]]/u[i]
                    if theta_i < theta :
                        theta = theta_i
                        p = i
            if theta == INF:
                self.log("simplex it {:2d}: Infinite ray found, unbounded problem".format(iteration),2)
                return False # infinite direction, z = -infinite

            # compute z
            z += theta*rq
            self.log("simplex it {:2d}: B({:d}) = {:2d} <-> {:2d}, theta = {:.3f}, \
z = {:.2f}".format(iteration, p, B[p], q, theta,z),2)
            # compute new feasible solution
            x[B] -= theta*u
            x[q] = theta
            x[B[p]] = 0
            B[p] = q  # replace basic variable by non basic
            # compute new Binv
            for i in range(len(u)):
                if i != p:
                    Binv[i,:] -= u[i]*Binv[p,:]/u[p]
            Binv[p,:] /= u[p]
            iteration += 1

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
        y = np.concatenate((np.zeros(self.N), self.b)) # SBF is B
        c = np.concatenate((np.zeros(self.N), np.ones(self.M)))
        self.log("Phase I: starting primal simplex with Bland's rule",1)
        found, x, B, z = self._simplex(y,c,B,Binv)
        if not found:
            self.log("Phase I: unbounded problem",1)
            return INF
        if z > 0:
            self.log("Phase I: optimal solution with z* = {:.2f} > 0".format(z),1)
            return -1; # Infactible


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
