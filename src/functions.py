from tqdm import tqdm
import os
import numpy as np
from numpy.ctypeslib import ndpointer
from collections import defaultdict
import ctypes
from scipy.linalg import svd
import scipy as sp
import scipy.stats


def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        text = inf.read().rstrip().split('\n')
        int_array = [line.split('\t')[1:] for line in text[1:]]

    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        text = inf.read().rstrip().split('\n')
        drug_sim = [line.rstrip().split('\t')[1:] for line in text[1:]]

    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        text = inf.read().rstrip().split('\n')
        target_sim = [line.rstrip().split('\t')[1:] for line in text[1:]]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        text = inf.read().rstrip().split('\n')
        drugs = text[0].split()
        targets = [line.strip("\n").split()[0] for line in text[1:]]
    return drugs, targets


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = int(index.size/num)
        for i in range(num):
            if i < num-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def svd_init(M, num_factors):
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


def makeResultFile(adjMatrix, dataset, fileInfo):
    with open(os.path.join('data', 'datasets', f'{dataset}_simmat_dc.txt'), 'w') as f:
        f.write('data\t')
        for file in fileInfo:
            f.write(file[0] + '\t')
        f.write('\n')

        for i in range(len(fileInfo)):
            f.write(fileInfo[i][0] + '\t')
            for j in range(len(fileInfo)):
                f.write('%0.5f\t' % adjMatrix[i][j])
            f.write('\n')


def getAdjMatrix(path, fileInfo, weights):
    dataset = path.split('/')[-1]

    print('\n------------------------')
    print('make: ', dataset)

    adjMatrix = [[0.0] * len(fileInfo) for _ in range(len(fileInfo))]

    icost, dcost, rcost = weights
    icost = float(icost)
    dcost = float(dcost)
    rcost = float(rcost)

    test_c_codes = ctypes.cdll.LoadLibrary("src/edit.so")

    c_fileInfo = []
    for info in fileInfo:
        c_fileInfo.append(info[1].encode())

    adj_c_func = test_c_codes.getAdjMatrix
    N = len(fileInfo)

    class struct(ctypes.Structure):
        _fields_ = [("fileInfo", ctypes.c_char_p * N)]

    t = struct()
    for i, info in enumerate(fileInfo):
        t.fileInfo[i] = info[1].encode()

    adj_c_func.restype = ndpointer(dtype=ctypes.c_double, shape=(N*N, ))

    adj_c_func.argtypes = [ctypes.POINTER(struct), ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    res = adj_c_func(ctypes.byref(t), N, icost, dcost, rcost)

    for i in tqdm(range(N)):
        for j in range(N):
            adjMatrix[i][j] = res[i*N + j]

    return adjMatrix
