import numpy as np


def qr_decom(matrix):
    k = np.array(matrix)
    k = k.astype('float64')
    c = np.copy(k)
    m, n = k.shape

    Q = np.zeros((m, n))

    for i in range(m):
        a = k[:, i]

        if (i > 0):
            for j in range(i):
                prev = Q[:, j]
                temp = k[:, i]
                a -= (np.dot(temp, prev)) * prev

        Q[:, i] = a / np.sqrt(np.dot(a, a))

    R = np.matmul(np.transpose(Q), c)

    return Q, R


def eigen(matrix, iteration):
    m = np.array(matrix)
    n = m.shape[0]
    eigenvector = np.identity(n)

    # pake qr sendiri
    """ for i in range(iteration):
        Q, R = qr_decom(matrix)
        eigenvector = np.matmul(eigenvector,Q)
        matrix = np.matmul(R,Q) """

    # pake qr numpy
    for i in range(iteration):
        m = np.copy(matrix)
        Q, R = np.linalg.qr(m)
        matrix = np.matmul(R, Q)
        eigenvector = np.matmul(eigenvector, Q)

    eigenvalue = np.diagonal(matrix)

    idxsort = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idxsort]
    eigenvector = eigenvector[:, idxsort]

    eigenvalue = np.around(eigenvalue, decimals=3)
    eigenvector = np.around(eigenvector, decimals=3)

    return eigenvalue, eigenvector

def eface(eigenvector, avrmatrix):
    eigenface = list()
    n = avrmatrix.shape[0]
    for i in range(n):
        eigenface.append(np.matmul(eigenvector, avrmatrix[0]))

    return eigenface

# buat testing
A = np.random.rand(20, 20)

inp = np.loadtxt('matrix.txt')

e = np.linalg.eig(inp)
eval = e[0]
evec = e[1]

eval = np.around(eval, decimals=3)
evec = np.around(evec, decimals=3)

print(eval)
print(evec)

result = eigen(inp, 100000) #iterationnya bebas diganti berapa
""" np.savetxt('eigval.txt',result[0])
np.savetxt('eigvec.txt',result[1]) """
print(result[0])
print(result[1])
