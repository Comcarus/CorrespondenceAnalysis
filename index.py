import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import pandas as pd


class CorrespondenceAnalysis(object):
    def __init__(self, ct):
        self.rows = ct.index.values if hasattr(ct, 'index') else None
        self.cols = ct.columns.values if hasattr(ct, 'columns') else None

        # Матрица сопряженности
        N = np.matrix(ct, dtype=float)
        # матрица соответствия из матрицы сопряженности
        P = N / N.sum()

        # предельные итоговые суммы строк и столбцов P в виде векторов
        r = P.sum(axis=1)
        c = P.sum(axis=0).T

        # диагональные матрицы сумм строк / столбцов
        D_r_rsq = np.diag(1. / np.sqrt(r.A1))
        D_c_rsq = np.diag(1. / np.sqrt(c.A1))

        # матрица остатков
        S = D_r_rsq * (P - r * c.T) * D_c_rsq

        U, D_a, V = svd(S, full_matrices=False)
        D_a = np.asmatrix(np.diag(D_a))
        V = V.T

        # координаты
        F = D_r_rsq * U * D_a
        G = D_c_rsq * V * D_a

        X = D_r_rsq * U
        Y = D_c_rsq * V

        self.F = F.A
        self.G = G.A
        self.X = X.A
        self.Y = Y.A

    def plot(self):
        xmin, xmax = None, None
        ymin, ymax = None, None
        if self.rows is not None:
            for i, t in enumerate(self.rows):
                x, y = self.F[i, 0], self.F[i, 1]
                plt.text(x, y, t, va='center', ha='center', color='r')
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.F[:, 0], self.F[:, 1], 'ro')

        if self.cols is not None:
            for i, t in enumerate(self.cols):
                x, y = self.G[i, 0], self.G[i, 1]
                plt.text(x, y, t, va='center', ha='center', color='b')
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.G[:, 0], self.G[:, 1], 'bs')

        if xmin and xmax:
            pad = (xmax - xmin) * 0.1
            plt.xlim(xmin - pad, xmax + pad)
        if ymin and ymax:
            pad = (ymax - ymin) * 0.1
            plt.ylim(ymin - pad, ymax + pad)

        plt.grid()
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')

def _test():
    df = pd.io.parsers.read_csv('data/fashion_brands.csv')
    df = df.set_index('brand')
    ca = CorrespondenceAnalysis(df)
    plt.figure(100)
    ca.plot()
    plt.show()


if __name__ == '__main__':
    _test()