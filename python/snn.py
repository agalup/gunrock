### sample python interface - pagerank

import sys, getopt
import ctypes
from ctypes import *
from numpy import *

#rmse
from pandas import DataFrame
import statsmodels.api as sm
import numpy as np

def Usage():
    print("python snn.py --market --labels=<filepath> --k=<int> --eps=<int> --min-pts=<int>")

def read_input(argv):
    labels_file = ''
    k = 0
    eps = 0
    min_pts = 0
    
    try:
        opts, args = getopt.getopt(argv, 'm:l:k:e:m', ['market=', 'labels=', 'k=', 'eps=', 'min-pts='])
    except getopt.GetoptError:
        Usage()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("--labels"):
            labels_file = arg
        elif opt == "--k":
            k = arg
        elif opt == "--eps":
            eps = arg
        elif opt == "--min-pts":
            min_pts = arg
    
    return(labels_file, k, eps, min_pts)

def read_labels(labels_file):
    datatest = open(labels_file, "r")
    lines = datatest.readlines()
    #print(lines[0])
    (n, dim) = lines[0].split()

    labels = np.full((int(dim), int(n)), 0, dtype=np.double)
    for i in range(1, int(n)):
        label = lines[i].split()
        for j in range(int(dim)):
            labels[j][i-1] = label[j]

    datatest.close()
    return (labels_file.encode('utf-8'), labels, n, dim)

def run_snn(gunrock, labels, n, dim, k, eps, min_pts):
    ### output data
    clusters = pointer((c_int * int(n))())
    clusters_counter = pointer(c_int(0))
    core_points_counter = pointer(c_int(0))
    noise_points_counter = pointer(c_int(0))
    k_ptr = pointer((c_int)(int(k)))
    eps_ptr = pointer((c_int)(int(eps)))
    min_pts_ptr = pointer((c_int)(int(min_pts)))

    gunrock_snn = gunrock.snn
    gunrock_snn.restype = ctypes.c_double
    
    ### call gunrock function on device
    elapsed = gunrock_snn(labels, k_ptr, eps_ptr, min_pts_ptr, 
            clusters, clusters_counter, core_points_counter, 
            noise_points_counter)
    
    ### sample results
    print ('Gunrock SNN call elapsed: ', double(elapsed))
    print ('number of clusters: ', clusters_counter.contents.value)
    print ('number of core points: ', core_points_counter.contents.value)
    print ('number of noise points: ', noise_points_counter.contents.value)

    range_ = 20
    if int(n) < range_:
        range_ = int(n)

    #for x in range(range_):
    #    print (x, ": ", clusters[0][x])

    # The number of the noise points checking
    np_counter = 0
    for x in range(int(n)):
        if clusters[0][x] < 0:
            np_counter += 1
    if np_counter != noise_points_counter.contents.value:
        print("The noise points number claimed by SNN is ", noise_points_counter.contents.value, ". Returned data shows", np_counter)
    return (clusters)

def compute_rmse(labels, clusters, n, dim):
    ### RMSE algo
    data = {}
    data['point'] = range(int(n))
    data['cluster_id'] = [clusters[0][x] for x in range(int(n))]
    data_columns = ['cluster_id']
    data_labels = []
    for label_id in range(int(dim)):
        data['label_'+str(label_id)] = labels[label_id]
        data_columns = np.append(data_columns, ['label_'+str(label_id)], axis=0)
        data_labels = np.append(data_labels, ['label_'+str(label_id)], axis=0)

    df = DataFrame(data, columns=data_columns)
    print(data['point'])
    print("data columns:", data_columns)
    print("data labelss:", data_labels)
    print("df data types:", df.dtypes)

    X = df[data_labels]
    Y = df['cluster_id']
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    print_model = model.summary()

    print(print_model)
    #print(predictions)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print("RMSE:" + str(np.sqrt(mean_squared_error(predictions, Y))))
    print("MAE:" + str(mean_absolute_error(predictions, Y)))

    limit = 20
    if int(n) < limit:
        limit = int(n)
    for x in range(limit):
        print(x, "prediction: ", predictions[x], "is: ", str(clusters[0][x]))


def main(argv):
    ### load gunrock shared library - libgunrock
    gunrock = cdll.LoadLibrary('../build/lib/libgunrock.so')
    (labels_file, k, eps, min_pts) = read_input(argv)
    (labels_path, labels, n, dim) = read_labels(labels_file)
    clusters = run_snn(gunrock, labels_path, n, dim, k, eps, min_pts)
    compute_rmse(labels, clusters, n, dim)

if __name__ == "__main__":
    main(sys.argv[1:])
