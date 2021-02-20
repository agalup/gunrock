### sample python interface - pagerank

import sys, getopt, os, time
import ctypes
from ctypes import *

from pandas import DataFrame
import statsmodels.api as sm
from numpy import *
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px
import plotly
import matplotlib.pyplot as plt
import seaborn as sns

def Usage():
    print("python snn.py --market=1 --labels=<filepath> --k=<int> --eps=<int> --min-pts=<int> --target=<filepath>")

def read_input(argv):
    labels_file = ''
    target_file = ''
    k = 0
    eps = 0
    min_pts = 0
    
    try:
        opts, _ = getopt.getopt(argv, 'm:l:k:e:m:t', ['market=', 'labels=', 'k=', 'eps=', 'min-pts=', 'target='])
    except getopt.GetoptError:
        print(getopt.GetoptError, "cause error")
        Usage()
        sys.exit(2)

    if len(opts) < 5:
        print("Number of arguments is less than 5: ", len(opts))
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
        elif opt == "--target" and arg != '':
            target_file = arg

    filename = os.path.basename(labels_file)
    name = os.path.splitext(filename)[0] 
    return(labels_file, target_file, name, k, eps, min_pts)

def read_labels(labels_file, target_file):
    traindata = open(labels_file, "r")
    lines = traindata.readlines()
    #print(lines[0])
    (n, dim) = lines[0].split()
    labels = np.full((int(n), int(dim)), 0, dtype=np.double)
    for i in range(1, int(n)):
        label = lines[i].split()
        for j in range(int(dim)):
            labels[i-1][j] = label[j]
    traindata.close()

    testdata = open(target_file, "r")
    lines = testdata.readlines()
    (n, dim) = lines[0].split()
    target = np.full((int(n), int(dim)), 0, dtype=np.int)
    for i in range(1, int(n)):
        label = lines[i].split()
        for j in range(int(dim)):
            target[i-1][j] = label[j]
    testdata.close()
    return (labels_file.encode('utf-8'), labels, target, int(n), int(dim))

def run_snn(gunrock, labels, n, dim, k, eps, min_pts):
    ### output data
    clusters = pointer((c_int * n)())
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
    
    # The number of the noise points checking
    np_counter = 0
    for x in range(n):
        if clusters[0][x] < 0:
            np_counter += 1
    if np_counter != noise_points_counter.contents.value:
        print("The noise points number claimed by SNN is ", noise_points_counter.contents.value, ". Returned data shows", np_counter)

    return (clusters, clusters_counter.contents.value, noise_points_counter.contents.value, core_points_counter.contents.value)

def assign_consecutive_labels(clusters, n):
    # Assigne consecutive numbers starting from 1 to the clusters with the exception of noise points
    #print (clusters)
    cluster_dict = {}
    np_counter = 1
    noise_points = False
    #clusters0 = np.full((1, int(n)), 0, dtype=int)
    clusters0 = np.full((int(n), 1), 0, dtype=int)
    for x in range(n):
        if clusters[0][x] > -1:
            if not clusters[0][x] in cluster_dict.keys():
                cluster_dict[clusters[0][x]] = np_counter
                clusters[0][x] = np_counter
                np_counter += 1
            else:
                clusters[0][x] = cluster_dict[clusters[0][x]]
        else:
            noise_points = True
        clusters0[x] = clusters[0][x]
    if noise_points:
        np_counter += 1
    #print("clusters ", cluster_dict)
    return (clusters0, np_counter-1)

def compute_rmse(labels, clusters, target, n, dim):
    print(labels.shape, clusters.shape, target.shape)
    if target.any():
        model = sm.OLS(target, labels).fit()
        predictions = model.predict(labels)
        print_model = model.summary()
        print("RMSE(clusters, target):" + str(np.sqrt(mean_squared_error(clusters, target))))
        print("RMSE(clusters, predictions):" + str(np.sqrt(mean_squared_error(clusters, predictions))))
        print("RMSE(target, predictions):" + str(np.sqrt(mean_squared_error(predictions, target))))
        #print("MAE:" + str(mean_absolute_error(predictions, )))
    
    model = sm.OLS(clusters, labels).fit()
    predictions = model.predict(labels)
    print_modle = model.summary()
    print("RMSE(clusters, predictions):" + str(np.sqrt(mean_squared_error(clusters, predictions))))
    print("MAE:" + str(mean_absolute_error(predictions, clusters)))

    limit = 20
    if n < limit:
        limit = n
    for x in range(limit):
        print(x, "prediction: ", predictions[x], "is: ", str(clusters[x]))

    return (np.sqrt(mean_squared_error(clusters, target) ), 
            np.sqrt(mean_squared_error(clusters, predictions)), 
            np.sqrt(mean_squared_error(predictions, target)))

def prepare_data(labels, clusters):
    X = labels
    y = clusters
    print(X.shape, y.shape)
    feat_cols = [ 'label_'+str(i) for i in range(X.shape[1]) ]

    df = DataFrame(labels, columns=feat_cols)
    print(df.shape)
    df['y'] = clusters
    print(df.shape)
    df['label'] = df['y'].apply(lambda i: str(i))
    print(df.shape)

    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    return (df, feat_cols, rndperm)

def visualization(df, rndperm, feat_cols, filename, num_clusters, dim):
    #reduce space dimension to 50
    n_components = 50
    if dim < n_components:
        n_components=dim

    #take a sampel of 10000 entries
    N = 10000
    if N > df.shape[0]:
        N = df.shape[0]
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values

    # counting the number of clusters in the subset data
    clusters_subset = df_subset['y'].values
    number_of_colors = 0
    cluster_dict = {} 
    for i in clusters_subset:
        if not i in cluster_dict.keys():
            cluster_dict[i] = i
            number_of_colors += 1
    print("number of colors: ", number_of_colors)
    
    # First compute PCA 50 components on data subset
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_subset)
    df_subset['pca-one'] = pca_result[:,0]
    if pca_result.any() > 1:
        df_subset['pca-two'] = pca_result[:,1]
        if pca_result.any() > 2:
            df_subset['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
  
    # TSNE 2 components on data subset
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(data_subset)
    print('t-SNE data subset done, elapsed time: {} seconds'.format(time.time()-time_start))
    df_subset['tsne-2d-one'] = tsne_pca_results[:,0]
    df_subset['tsne-2d-two'] = tsne_pca_results[:,1]

    # TSNE 2 components on pca_result
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result)
    print('t-SNE pca_result done, elapsed time: {} seconds'.format(time.time()-time_start))

    tsne_pca_one = 'tsne-pca'+str(n_components)+'-one'
    tsne_pca_two = 'tsne-pca'+str(n_components)+'-two'
    df_subset[tsne_pca_one] = tsne_pca_results[:,0]
    df_subset[tsne_pca_two] = tsne_pca_results[:,1]

    fig2 = plt.figure(figsize=(16,10))
    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="cluster_id",
        palette=sns.color_palette("hls", n_colors=number_of_colors),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="cluster_id",
        palette=sns.color_palette("hls", n_colors=number_of_colors),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax2
    )
    ax3 = plt.subplot(1, 3, 3)
    sns.scatterplot(
        x=tsne_pca_one, y=tsne_pca_two,
        hue="cluster_id",
        palette=sns.color_palette("hls", n_colors=number_of_colors),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax3
    )
    fig2.savefig(filename+'_pca'+str(n_components)+'_tsne2'+'.png')
    

def main(argv):
    ### load gunrock shared library - libgunrock
    gunrock = cdll.LoadLibrary('../build/lib/libgunrock.so')
    ### read input variables for SNN: labels file path, k, eps, min-pts
    (labels_file, target_file, filename, k, eps, min_pts) = read_input(argv)
    ### read labels, n, dim
    (labels_path, labels, target, n, dim) = read_labels(labels_file, target_file)
    ### run SNN Gunrock
    (clusters, nr_clusters, nr_noise, nr_cp) = run_snn(gunrock, labels_path, n, dim, k, eps, min_pts)
    ### assign consecutive cluster labels except of noise points
    (clusters, num_clusters) = assign_consecutive_labels(clusters, n)
    ### run Ordinary Least Squares estimation, compute RMSE
    (r1, r2, r3) = compute_rmse(labels, clusters, target, n, dim)
    ### conver matrix and vector to a pandas dataframe
    (df, feat_cols, rndperm) = prepare_data(labels, clusters) 
    output = open("result/"+filename+'_'+str(k)+'_'+str(eps)+'_'+str(min_pts), "w")
    output.write("num_clusters: " + str(nr_clusters)+ "\n")
    output.write("num_noises:   " + str(nr_noise)+    "\n")
    output.write("num_core_points:" + str(nr_cp) +    "\n")
    output.write("RMSE(clusters, target):" + str(r1)+   "\n")
    output.write("RMSE(clusters, predictions):" + str(r2) +"\n")
    output.write("RMSE(target, predictions):" + str(r3) +  "\n")
    idx =0
    for x in clusters:
        output.write("clusters("+ str(idx)+ ")="+ str(x)+ "\n")
        idx += 1
    output.close()
    ### run TSNE
    ##visualization(df, rndperm, feat_cols, filename+'_'+str(k)+'_'+str(eps)+'_'+str(min_pts), num_clusters, dim)

if __name__ == "__main__":
    main(sys.argv[1:])
