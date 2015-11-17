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
        self.verbose = verbose
        self.readFile(filename)
        self.M, self.N = self.A.shape # A is M x N matrix
        self.max_it = 200


    def readFile(self, filename) :
        self.c, self.A, self.b = [np.loadtxt(A) for A in generateChunks(filename, 3)]
        self.log("SIMPLEX: Have read c,A,b from %s" % filename)

    # return entring basic variable, or -1 if no negative reduced cost (optimal)
    def _blandRule(self, c, B, Binv) :
        l = np.dot(c[B], Binv) # cB*Binv = lambda from dual
        # generator to get reduced costs (computes on the fly)
        r = ( (j, c[j] - np.inner(l,self.A[:,j])) for j in range(self.A.shape[1]) if j not in B)
        # return (rj, j) in increasing order
        rj = next(r) # blandRule -> pick first < 0
        while rj != None and rj[1] >= 0 :
            rj = next(r,None) # return None at end
        return (-1,0) if rj == None else rj # rj == None => rj >= 0 for all j => optimal

    # change column col of Basis B with u, compute new inverse from previous one
    def _recomputeBinv(self, col, u, Binv) :
        for i in range(len(u)):
            if i != col : Binv[i,:] -= u[i]*Binv[col,:]/u[col]
        Binv[col,:] /= u[col]

    # apply simplex algorithm given a basis, its inverse, and a basic feasible solution
    def _simplex(self, x, c, B, Binv):
        iteration = 1
        z = np.inner(c,x) # first computation of z
        self.log("simplex it 0, starting with z = {}".format(z),2)
        while iteration <= self.max_it : # avoid infinite loop
            q , rq = self._blandRule(c, B, Binv) # get q, entring basic variable
            if q == -1 : # => optimal
                self.log("simplex it {:2d}: Optimal solution found, z* = {:.2f}".format(iteration, z),2)
                return True, x, B, np.inner(c,x) # x is optimal with basis B, recompute z for precision
            # compute directions u = -dq = Binv*A[q]
            u = np.dot(Binv, self.A[:,q])
            # select B(p) leaving variable
            theta = INF # infinite at first
            p = -1
            for i in range(len(u)):
                if u[i] > 0 : # for positive components (negative for dq)
                    theta_i = x[B[i]]/u[i]
                    if theta_i < theta :
                        theta = theta_i
                        p = i
            if theta == INF: # => all directions are non-positive => -d >= 0
                self.log("simplex it {:2d}: Infinite ray found, unbounded problem".format(iteration),2)
                return False, x, B, -INF # infinite direction, z = -infinite
            # compute new z
            z += theta*rq
            self.log("simplex it {:2d}: B({:d}) = {:2d} <-> {:2d}, theta = {:.3f}, \
z = {:.2f}".format(iteration, p, B[p], q, theta,z),2)
            # compute new feasible solution
            x[B] -= theta*u # move along direction
            x[q] = theta
            x[B[p]] = 0 # put a real 0 to fix epsilons
            B[p] = q  # replace basic variable in basis
            self._recomputeBinv(p, u, Binv) # compute new Binv after changing column p of Binv by u
            iteration += 1

    # compute a first BFS for our problem introducing artificial variables
    def _phaseI(self):
        self.log("SIMPLEX: Phase I started")
        self.log("Phase I: make b >= 0", 1)
        # make independent term positive, so that y >= 0
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i,:] *= -1
                self.b[i] *= -1
        self.log("Phase I: add auxiliary variables",1)
        B = -np.ones(self.M, dtype=np.int) # init basis with -1 (undefined)
        # find columns with e_i
        for j in range(self.N) :
            i = np.argmax(self.A[:,j])
            # norm1 column j == 1 and A[i,j] = 1 => column Aj is e_i
            if np.linalg.norm(self.A[:,j], ord=1) == 1 and self.A[i,j] == 1 and B[i] == -1 :
                B[i] = j
        # complete missing e_j of Basis
        for j in range(self.M) :
            if B[j] == -1 : # add e_j at end
                e_j = np.zeros(self.M)
                e_j[j] = 1
                self.A = np.c_[self.A, e_j]
                B[j] = self.A.shape[1]-1
        Binv = np.eye(self.M) # inverse of identity is indentity
        # create y and c
        y = np.zeros(self.A.shape[1])
        c = np.zeros(self.A.shape[1])
        y[B] = self.b # trivial BFS
        c[B] = 1 # cost in 1 for y and 0 for x
        self.log("Phase I: starting primal simplex with Bland's rule",1)
        found, x, B, z = self._simplex(y,c,B,Binv)
        if not found:
            self.log("Phase I: unbounded problem",1)
            return False, INF
        if z > 0: # => no BFS existent => empty polyhedron
            self.log("Phase I: optimal solution with z* = {:.2f} > 0".format(z),1)
            return False, -1 # => infeasible
        self.log("Phase I: remove artificial variables",1)
        # remove columns of artificial variables
        if np.linalg.norm(x[self.N:]) != 0 : self.log("ERROR: artificial variables should be 0")
        self.A = self.A[:,:self.N] # remove added artificial variables
        x = x[:self.N]
        self.log("Phase I: end")
        return True, x, B, Binv

    # apply simplex algorithm, phase I and II with Bland's rule
    def solve(self):
        ret = self._phaseI() # phase I
        if not ret[0]: #
            if ret[1] == INF:
                self.log("SIMPLEX: unbounded problem, z* = -inf")
            else:
                self.log("SIMPLEX: infeasible problem")
        else:
            x, B, Binv = ret[1:] # unpack result
            self.log("SIMPLEX: Phase II started")
            self.log("Phase II: starting simplex primal with Bland's rule",1)
            found, x, B, z = self._simplex(x, self.c, B, Binv)
            if not found:
                self.log("SIMPLEX: unbounded problem z* = -inf")
            else:
                self.log("Phase II: end",1)
                self.log("SIMPLEX: optimal solution found")
                np.set_printoptions(precision=1, suppress=True) # output with 1 decimal, no scientific notation
                self.log("B = {}".format(B))
                self.log("Xb = {}".format(x[B]))
                self.log("Z* = {:.4f}".format(z))
                # compute reduced costs
                mask = np.ones(len(x), dtype=bool) # mask to get only basic variables
                mask[B] = 0
                r = self.c[mask] - np.dot(np.dot(self.c[B],Binv), self.A[:,mask])
                self.log("r = {}".format(r))
        self.log("SIMPLEX: end\n")

    # display A,b,c basic parameters of problem
    def display(self):
        print("VECTOR C:")
        print(self.c)
        print("MATRIX A:")
        print(self.A)
        print("VECTOR B:")
        print(self.b)
    # print output in verbose mode
    def log(self, msg, indent=0):
        if self.verbose: print('\t'*indent + msg)

def main(argv):
    if len(argv) < 2:
        print("Usage: ./simplex.py data1 data2 ... dataN\nNeeds at least 1 data file")
        sys.exit(1)
    # We have at least one data
    for i in range(1,len(argv)) : # apply simplex for each one of them
        print("------ Data input %d ------" % i)
        S = Simplex(argv[i], verbose=True)
        S.solve();

main(sys.argv)
