from sympy import *
import galois
import numpy as np
from tqdm import tqdm 
import random

p = 167
assert isprime(p), "p is not prime"
assert (p % 8 == 1) or (p % 8 == 7), "p is not 1 or -1 mod 8"

n = p + 1


def qr_set(p):
    qr_set = set({})
    i = 1
    while len(qr_set) < (p+1)/2:
        qr_set.add(i**2 % p)
        i += 1
    qr_set.remove(0)
    return qr_set

GF = galois.GF(2)
G = GF(np.zeros((n,n), dtype = int))
for k in range(n):
    G[n-1, k] = 1
    G[k, n-1] = (p % 8 == (p-1))
for i in range(n-1):
    for j in range(n-1):
        G[i,j] = (j-i)%p in qr_set(p)

def is_row_echelon_form(matrix):
    if not matrix.any():
        return False

    rows = matrix.shape[0]
    cols = matrix.shape[1]
    prev_leading_col = -1

    for row in range(rows):
        leading_col_found = False
        for col in range(cols):
            if matrix[row, col] != 0:
                if col <= prev_leading_col:
                    return False
                prev_leading_col = col
                for r in range(row):
                    if matrix[r, col] != 0:
                        return False
                leading_col_found = True
                break
        if not leading_col_found and any(matrix[row, col] != 0 for col in range(cols)):
            return False
    return True

def find_nonzero_row(matrix, pivot_row, col):
    nrows = matrix.shape[0]
    for row in range(pivot_row, nrows):
        if matrix[row, col] != 0:
            return row
    return None

# Swapping rows so that we can have our non zero row on the top of the matrix
def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]

def make_pivot_one(matrix, pivot_row, col):
    pivot_element = matrix[pivot_row, col]
    matrix[pivot_row] //= pivot_element
    # print(pivot_element)

def eliminate(matrix, pivot_row, col):
    nrows = matrix.shape[0]
    pivot_element = matrix[pivot_row, col]
    for row in [x for x in range(nrows) if x != pivot_row]:
        factor = matrix[row, col]
        matrix[row] -= factor * matrix[pivot_row]

# Implementing above functions
def row_echelon_form(matrix):
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    pivot_row = 0
# this will run for number of column times. If matrix has 3 columns this loop will run for 3 times
    for col in range(ncols):
        nonzero_row = find_nonzero_row(matrix, pivot_row, col)
        if nonzero_row is not None:
            swap_rows(matrix, pivot_row, nonzero_row)
            make_pivot_one(matrix, pivot_row, col)
            eliminate(matrix, pivot_row, col)
            pivot_row += 1
    return matrix


matrix = G
#print("Matrix Before Converting:")
#print(matrix)
#print()
result = row_echelon_form(matrix)
#print("After Converting to Row Echelon Form:")
#print(result)
#if is_row_echelon_form(result):
#    print("In REF")
#else:
#    print("Not in REF--------------->")

result = result[~np.all(result == 0, axis=1)]
#print(result)
C = result[:, :-1]
n = C.shape[1]
k = C.shape[0]
A = C[:, -(n-k):]
C_orth = np.concatenate((A.T, GF(np.identity(n-k, dtype = int), dtype=int)), axis=1, dtype = int)
#print(C_orth)

#swept = row_echelon_form(C_orth)
#flipped = np.flip(swept, axis = 0)
#print(flipped)
def distance(C):
    d = k
    m = min(2**k, 10**7)
    for i in tqdm(range(m, 2*m)):
        z = np.zeros(k, dtype = int)
        y = np.array([int(x) for x in bin(i)[2:]])
        z[-len(y):] = y
        z = z.reshape(-1, 1)
        #print(z)
        z = GF(z, dtype = int)
        word = np.matmul(z.T,C)
        #print(word)
        weight = np.count_nonzero(word)
        d = min(d, weight)
    return d
#distance(C)
print(distance(row_echelon_form(C)))