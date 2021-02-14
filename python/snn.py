### sample python interface - pagerank

import sys, getopt
from ctypes import *

def Usage():
    print("python snn.py --market --labels=<filepath> --k=<int> --eps=<int> --min-pts=<int>")

def read_input(argv):
    labels = ''
    k = 0
    eps = 0
    min_pts = 0
    n = 0
    dim = 0

    try:
        opts, args = getopt.getopt(argv, 'm:l:k:e:m', ['market=', 'labels=', 'k=', 'eps=', 'min-pts='])
    except getopt.GetoptError:
        Usage()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("--labels"):
            labels_file = arg
            datatest = open(labels_file, "r")
            line = datatest.readline()
            (n, dim) = line.split()
            datatest.close()
            
            ### input data
            labels = labels_file.encode('utf-8')

        elif opt == "--k":
            k = arg
        elif opt == "--eps":
            eps = arg
        elif opt == "--min-pts":
            min_pts = arg
    
    return(labels, n, dim, k, eps, min_pts)

def run_snn(gunrock, labels, n, dim, k, eps, min_pts):
    ### output data
    clusters = pointer((c_int * int(n))())
    clusters_counter = pointer(c_int(0))
    core_points_counter = pointer(c_int(0))
    noise_points_counter = pointer(c_int(0))
    k_ptr = pointer((c_int)(int(k)))
    eps_ptr = pointer((c_int)(int(eps)))
    min_pts_ptr = pointer((c_int)(int(min_pts)))
    
    ### call gunrock function on device
    elapsed = gunrock.snn(labels, k_ptr, eps_ptr, min_pts_ptr, 
            clusters, clusters_counter, core_points_counter, 
            noise_points_counter)
    
    ### sample results
    print ('elapsed: ', elapsed)
    print ('number of clusters: ', clusters_counter.contents.value)
    print ('number of core points: ', core_points_counter.contents.value)
    print ('number of noise points: ', noise_points_counter.contents.value)

def main(argv):
    ### load gunrock shared library - libgunrock
    gunrock = cdll.LoadLibrary('../build/lib/libgunrock.so')
    (labels, n, dim, k, eps, min_pts) = read_input(argv)
    run_snn(gunrock, labels, n, dim, k, eps, min_pts)
    

if __name__ == "__main__":
    main(sys.argv[1:])
