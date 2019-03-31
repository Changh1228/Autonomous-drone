from __future__ import print_function

import sys
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple


parser = ArgumentParser()
parser.add_argument('ground_truth')
parser.add_argument('estimation')


class Row(namedtuple('RowBase', 'sign tx ty tz qx qy qz qw')):
    @property
    def t(self):
        return np.r_[self.tx, self.ty, self.tz]


def loadfile(fn):
    with open(fn) as f:
        L = []
        for line in f:
            cols = line.strip().replace(',', '\t').split('\t')
            L.append(Row(cols[0], *map(float, cols[1:1+3+4])))
        L.sort(key=lambda row: (row.sign, np.linalg.norm((row.tx, row.ty, row.tz))))
        return L


def align_umeyama(model, data, rescale=True):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    aligned = s * R * data + t

    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))
    s = 1.0/sigma2*np.trace(np.dot(D_svd, S)) if rescale else 1.0
    t = mu_M-s*np.dot(R, mu_D)

    return (s * np.dot(R, data.T)).T + t


def main(threshold=1.0):
    args = parser.parse_args()
    ground_truth = loadfile(args.ground_truth)
    estimation   = loadfile(args.estimation)

    ground_truth_pos = np.array([(row.tx, row.ty, row.tz) for row in ground_truth])
    estimation_pos   = np.array([(row.tx, row.ty, row.tz) for row in estimation])

    # Find the subset of the ground truth objects that is in the estimated set.
    ground_truth_paired = []
    for est_row in estimation:
        try:
            i, d = min([(i, np.linalg.norm(est_row.t - gt_row.t))
                        for i, gt_row in enumerate(ground_truth)
                        if est_row.sign == gt_row.sign],
                       key=lambda pair: pair[1])
        except ValueError:
            #print('No match for estimated position of sign type', est_row.sign)
            continue
        else:
            ground_truth_paired.append(ground_truth.pop(i))

    del ground_truth

    if len(ground_truth_paired) < len(estimation):
        print('\x1b[31;1m', end='')
        print('! ! ! ESTIMATION SET LARGER THAN GROUND THRUTH SET, CHEATING DETECTED ! ! !')
        print('! ! !   PLEASE REMAIN SEATED WHILE SECURITY ARRIVE AT YOUR LOCATION   ! ! !')
        print('\x1b[0m', end='')
        sys.exit(1)

    model = np.array([(row.tx, row.ty, row.tz) for row in ground_truth_paired])
    data  = np.array([(row.tx, row.ty, row.tz) for row in estimation])

    align_se3 = align_umeyama(model, data, rescale=False)
    error_norm = np.linalg.norm(align_se3 - model, axis=1)
    within_tol = error_norm < threshold
    print('\x1b[32;1mMATCHES:\x1b[0m', np.count_nonzero(within_tol))

    align_sim3 = align_umeyama(model, data)
    error = align_sim3 - model
    print('\x1b[32;1mMSE:\x1b[0m    ', np.mean(np.linalg.norm(error, axis=1)))


if __name__ == '__main__':
    main()
