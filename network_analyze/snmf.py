import numpy as np


class SNNMF:

    max_iters = 100

    def fit(self, X, Y, r, lambda_):
        eps = np.power(0.1, 10)
        A = np.random.rand(X.shape[0], r)
        B = np.random.rand(Y.shape[0], r)
        S = np.random.rand(r, X.shape[1])
        # LY = np.multiply(L, Y)
        for i in range(self.max_iters):
            A = np.multiply(A, np.divide(X @ S.T, A@S@S.T+eps))
            B = np.multiply(B, np.divide(Y@S.T, B@S@S.T+eps))
            licznik = A.T@X + lambda_*B.T@Y
            mianownik = A.T@A@S + lambda_*B.T@ B@S
            S = np.multiply(S, (np.divide(licznik, mianownik + eps)))

        return A, B, S


if __name__ == '__main__':
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    Y = np.array([[1, 0], [0, 0], [0, 1]])
    snmf = SNNMF()
    A, B, S = snmf.fit(X, Y, 2, 5)
    print(S)