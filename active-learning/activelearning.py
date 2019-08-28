# -*- coding: utf-8 -*-

"""
Created on Fri Apr  5 13:48:04 2019

@author: Guillermo Climent, Ruben Gimenez & Mayra Russo
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import resample
from our_al_base import ao, show_results
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
# Import RBF kernel
from sklearn.metrics.pairwise import rbf_kernel

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


SEED = 666
np.random.seed(SEED)

# Data Import
train_labeled_path = "data/semeion_labeled.csv"
train_unlabaled_path = "data/semeion_unlabeled.csv"
test_path = "data/semeion_test.csv"

train_labeled = pd.read_csv(train_labeled_path, header = None, delimiter = ",")
train_unlabeled = pd.read_csv(train_unlabaled_path, header = None, delimiter = ",")

test = pd.read_csv(test_path, header = None, delimiter = ",")

# image sample viewing
image_train_unlabeled  = train_unlabeled.iloc[5,:].values[1:]
image_train_unlabeled.reshape(16,16)
plt.imshow(image_train_unlabeled.reshape(16,16))

## TRAIN ##
# split into data and labels
labeled_data = train_labeled.iloc[:, 1:] .to_numpy()
labels = train_labeled.iloc[:,0].to_numpy()

unlabeled_data = train_unlabeled.iloc[:, 1:].to_numpy()
labels_u = train_unlabeled.iloc[:, 0].to_numpy() # do NOT use

## TEST ##
test_data  = test.iloc[:,1:].to_numpy()
test_labels = test.iloc[:,0].to_numpy()

###FUNCTIONS###

## Diversity Methods ##
def MAO_Diversity(idx):
    # Lets limit the pool of possible samples as:
    idx = idx[0:query_points * 10]
    # Measure distances using the kernel function
    K = rbf_kernel(active.xunlab, gamma=active.gamma)
    Sidx = np.zeros(query_points, dtype=type(idx[0]))
    for j in np.arange(query_points):
        # Add the first point (and remove it from pool)
        Sidx[j] = idx[0]
        idx = idx[1:]
        # Compute distances (kernel matrix)
        # Distances between selected samples (Sidx) and the rest (idx)
        Kdist = K[Sidx[0:j+1],:][:,idx]
        # Obtain the minimum distance for each column
        Kdist = Kdist.min(axis=0)
        # Re-order by distance
        idx = idx[Kdist.argsort(axis=0)]
    # Move selected samples from unlabeled set to labeled set
    return active.updateLabels(Sidx)

def MAO_lambda_Diversity(idx, yp, ssc_method = "none", lam = 0.6):
    # MAO lambda: trade-off between uncertainty and diversity
    K = rbf_kernel(active.xunlab, gamma = active.gamma) #provisional kernel
    Sidx = np.zeros(query_points, dtype=type(idx[0]))
    for j in np.arange(query_points):
        # Add the first point, and remove it from pool
        Sidx[j] = idx[0]
        idx = idx[1:]
        # Compute distances (kernel matrix)
        # Distances between selected samples (Sidx) and the rest (idx)
        Kdist = np.abs(K[Sidx[0:j+1],:][:,idx])
        # Obtain the minimum distance for each column
        Kdist = Kdist.min(axis=0)
        # Trade-off between AL algorithm and Diversity
        if ssc_method == "ssc":
            heuristic = yp[idx, -1] * lam + Kdist * (1 - lam)
        elif ssc_method == "none":
            heuristic = yp[idx] * lam + Kdist * (1 - lam)
        idx = idx[heuristic.argsort()]  # axis=0
    # Move selected samples from unlabeled set to labeled set
    return active.updateLabels(Sidx)

def Cluster_Diversity(idx):
    # Init kmeans
    kmeans = KMeans(n_clusters=min(query_points, len(active.xunlab[idx])), random_state = SEED)
    # 1. Cluster the unlabeled set
    clusterIDs = kmeans.fit_predict(active.xunlab[idx])
    # 2. Select one sample per cluster.
    Sidx = np.zeros((query_points,), dtype=int)
    for j in np.arange(query_points):
        Sidx[j] = idx[clusterIDs == j][0]
    # Move selected samples from unlabeled set to labeled set
    return active.updateLabels(Sidx)


## Active Learning Algorithms ##
def Random_Sampling(xlab, ylab, xunlab):
    # 1. Fit random model
    random.classifier.fit(xlab, ylab)
    # 2. Choose some random samples from xunlab, yunlab
    idx = np.random.permutation(xunlab.shape[0])
    # 3. Call updateLabels to move them from pool to train
    return random.updateLabels(idx[0:query_points])

def MarginSampling(xlab, ylab, xunlab, div_method):
    active.classifier.fit(xlab, ylab)
    # Use heuristic to choose samples from pool
    dist = np.abs(active.classifier.decision_function(active.xunlab))
    dist = np.sort(dist, axis = 1)
    dist = dist[:, 0] #why
    idx = np.argsort(dist)
    # determine diversity method
    if div_method == "none":
        return  active.updateLabels(idx[0:query_points])
    elif div_method == "mao_div":
        return MAO_Diversity(idx = idx)
    elif div_method == "mao_lambda_div":
        return MAO_lambda_Diversity(idx = idx, yp = dist)
    elif div_method == "clustering_div":
        return Cluster_Diversity(idx = idx)

def MCLU(xlab, ylab, xunlab, div_method):
    active.classifier.fit(active.xlab, active.ylab)
    # Use heuristic to choose samples from pool
    dist = np.abs(active.classifier.decision_function(xunlab))
    dist = np.sort(dist, axis=1)
    dist = dist[:,-1] - dist[:,-2]
    idx = np.argsort(dist)
    # determine diversity method
    if div_method == "none":
        return active.updateLabels(idx[0:query_points])
    elif div_method == "mao_div":
        return MAO_Diversity(idx = idx)
    elif div_method == "mao_lambda_div":
        return MAO_lambda_Diversity(idx = idx, yp = dist)
    elif div_method == "clustering_div":
        return Cluster_Diversity(idx = idx)

def SSC(xlab, ylab, xunlab, div_method):
    model = DecisionTreeClassifier(random_state = SEED)
    active.classifier.fit(xlab, ylab)
    xtr = xlab
    ytr = ylab
    ytr = np.zeros(ytr.shape)
    ytr[active.classifier.support_] = 1
    if len(np.unique(ytr)) == 1:
        idx = np.random.permutation(xunlab.shape[0])
    else:
        model.fit(xtr, ytr)
        possible_SVs = model.predict(xunlab)
        idx = np.arange(xunlab.shape[0])[possible_SVs == 1]
        idx = np.random.permutation(idx)
    # determine diversity method
    if div_method == "none":
        return active.updateLabels(idx[0:query_points])
    elif div_method == "mao_div":
        return MAO_Diversity(idx = idx)
    elif div_method == "mao_lambda_div":
        dist = np.abs(active.classifier.decision_function(xunlab))
        return MAO_lambda_Diversity(idx = idx, yp = dist, ssc_method = "ssc")
    elif div_method == "clustering_div":
        return Cluster_Diversity(idx = idx)

def nEQB(xlab, ylab, xunlab, div_method,n_models = 4):
    # Number of models
    n_models = n_models
    # Base classifier for committee
    model = SVC(C=active.C, gamma=active.gamma)
    active.classifier.fit(xlab, ylab)
    # Use heuristic to choose samples from pool
    n_unlab = xunlab
    n_unlab = n_unlab.shape[0]
    predMatrix = np.zeros((n_unlab, n_models))
    for j in range(n_models):
        # Replica bootstrap
        while True:
            xbag, ybag = resample(xlab, ylab, replace=True)
            # Ensure that we have all classes in the bootstrap replica
            if len(np.unique(ybag)) >= n_classes:
                break
        model.fit(xbag, ybag)
        predMatrix[:, k] = model.predict(xunlab)
    # Count number of votes per class
    ct = np.zeros((xunlab.shape[0], n_classes))
    for i, w in enumerate(classes):
        ct[:, i] = np.sum(predMatrix == w, axis=1)
    ct /= n_models
    Hbag = ct.copy()
    # Set to 1 where Hbag == 0 to avoid -Inf and NaNs (problem is that 0 * -Inf = NaN)
    Hbag[Hbag == 0] = 1
    Hbag = -np.sum(ct * np.log(Hbag), axis=1) #este pal lambda
    logNi = np.log(np.sum(ct > 0, axis=1))
    # Avoid division by zero
    logNi[logNi == 0] = 1
    nEQB = Hbag / logNi
    # Select randomly one element among the ones with maximum entropy
    idx = np.where(nEQB == np.max(nEQB))[0]
    np.random.shuffle(idx)
    # determine diversity method
    if div_method == "none":
        return active.updateLabels(idx[0:query_points])
    elif div_method == "mao_div":
        return MAO_Diversity(idx = idx)
    elif div_method == "mao_lambda_div":
        return MAO_lambda_Diversity(idx = idx, yp = Hbag)
    elif div_method == "clustering_div":
        return Cluster_Diversity(idx = idx)



"""
EXERCISE 1

USE THIS LOOP TO TEST FUNCTIONS
algorithm options : MarginSampling, MCLU, SCC & nEQB

diversity methods (div_method): "none", "mao_div", "mao_lambda_div" & "clustering_div"
"""
# AL Variables
num_queries = 40
query_points = 10

classes = np.unique(labels)
n_classes = len(classes)

# Setup random and active objects
random = ao()
xtest, ytest = random.setup(labeled_data, labels, unlabeled_data, labels_u, test_data, test_labels)
white_class = random.copy()
# active = white_class.copy()

# Packing AL Algorithms % Diversity Criterions
algorithms = [MarginSampling, MCLU, SSC,  nEQB]
div_methods = ["none", "mao_div", "mao_lambda_div", "clustering_div"]


# Saving average accuracies & 2*sd
average_accuracies = pd.DataFrame()
alg_names = []
alg_div_names = []
avg_acc = []
std2 = []

for d in range(len(div_methods)):
    for k in range(len(algorithms)):
        # Update class instances for each AL algorithm iteration
        active = white_class.copy()
        random = white_class.copy()
        # Save Accuracy for each method
        accr_list = []
        acca_list = []
        for i in np.arange(0, num_queries):
            # print(i)
            # Random Method
            Random_Sampling(random.xlab, random.ylab, random.xunlab)
            accr = random.score(xtest, ytest)
            accr_list.append(accr)

            # AL
            algorithm = algorithms[k]
            algorithm(active.xlab, active.ylab, active.xunlab, div_method = div_methods[d])
            acca = active.score(xtest, ytest)
            acca_list.append(acca)
        # Packing accuracies into a DataFrame
        accuracies = pd.DataFrame()
        accuracies[algorithms[k].__name__] = acca_list
        accuracies["Random"] = accr_list
        # Printing Accuracy Results for itetations 10, 25, 50 & 70
        print("Algorithm:", algorithms[k].__name__," | ","Diversity Criterion:", div_methods[d])
        print("--------------------------------------------------------------")
        for quer in [4, 9, 14, 19]:
            print(quer+1, "queries:")
            print("Random ACC: %6.3f, Active ACC: %6.3f" % (accuracies.iloc[quer,1], accuracies.iloc[quer,0]))
        # Saving positions where non-random algorithm overcomes the random one
        aux = accuracies[accuracies[algorithms[k].__name__] > accuracies["Random"]]
        print("\nTimes", algorithms[k].__name__, "overcomes Random Sampling: ", aux.shape[0])
        # Printing in which n of queries AL algorithm performs better
        if aux.shape[0] == 0:
            aux1 = accuracies.sort_values([algorithms[k].__name__], ascending=False)
            print("Highest performance of", algorithms[k].__name__, "with", aux1.index[0]+1, "queries")
            aux1 = aux1.reset_index(drop=True)
            print("Random ACC: %6.3f, Active ACC: %6.3f" % (aux1.iloc[0,1], aux1.iloc[0,0]))
        else:
            aux1 = aux.sort_values([algorithms[k].__name__], ascending=False)
            print("Highest performance of", algorithms[k].__name__, "with", aux1.index[0]+1, "queries")
            aux1 = aux1.reset_index(drop=True)
            print("Random ACC: %6.3f, Active ACC: %6.3f" % (aux1.iloc[0,1], aux1.iloc[0,0]))
        # Plotting Accuracy Graphs
        plt.figure(figsize=(9, 5))
        plt.suptitle("Accuracy Scores, Random Sampling VS " + algorithms[k].__name__, fontsize=16)
        plt.title("Diversity Method: " + div_methods[d], fontsize=12)
        plt.xlabel("Number of Queries")
        plt.ylabel("Accuracy")
        ax = sns.lineplot(data=accuracies, palette="viridis")
        ax.legend(loc="lower right")
        plt.show()
        print("\n")


"""
EXERCISE 2

Repeating the experiment

algorithm options : MarginSampling, MCLU, SCC & nEQB

diversity methods (div_method): "none", "mao_div", "mao_lambda_div" & "clustering_div"
"""
# AL Variables
num_queries = 40
query_points = 10
num_of_experiments = 4

classes = np.unique(labels)
n_classes = len(classes)

# Packing AL Algorithms % Diversity Criterions
algorithms = [MarginSampling, MCLU, SSC,  nEQB]

div_methods = ["none", "mao_div", "mao_lambda_div", "clustering_div"]

# Saving average accuracies & 2*sd
AL_information = pd.DataFrame()
alg_names = []
alg_div_names = []
random_acc = []
al_acc = []
std2 = []
experiment_row_list = []
queries_row_list = []

for exp in range(num_of_experiments):
    # Set up random and active objects inside the experiment loop
    random = ao()
    xtest, ytest = random.setup(labeled_data, labels, unlabeled_data, labels_u, test_data, test_labels)
    white_class = random.copy()
    experiment = exp+1
    for d in range(len(div_methods)):
        for k in range(len(algorithms)):
            # Update class instances for each AL algorithm iteration
            active = white_class.copy()
            random = white_class.copy()
            # Save Accuracy for each method
            accr_list = []
            acca_list = []
            for i in np.arange(0, num_queries):
                # Random Method
                Random_Sampling(random.xlab, random.ylab, random.xunlab)
                accr = random.score(xtest, ytest)
                accr_list.append(accr)
                # AL
                algorithm = algorithms[k]
                algorithm(active.xlab, active.ylab, active.xunlab, div_method = div_methods[d])
                acca = active.score(xtest, ytest)
                acca_list.append(acca)

            # Information of every algorithm & experiment must be saved up
            experiment_row_list.extend(np.repeat(experiment, len(acca_list)))
            queries_row = np.arange(1,num_queries+1)
            queries_row_list.extend(queries_row)
            alg_names.extend(np.repeat(algorithms[k].__name__, len(acca_list)))
            alg_div_names.extend(np.repeat(div_methods[d], len(acca_list)))
            random_acc.extend(accr_list)
            al_acc.extend(acca_list)
    # if the loop is in its last iteration, print results:
    if exp == num_of_experiments-1:
        AL_information["Experiment"] = experiment_row_list
        AL_information["nQueries"] = queries_row_list
        AL_information["Algorithm"] = alg_names
        AL_information["Diversity_Method"] = alg_div_names
        AL_information["AL_Accuracy"] = al_acc
        AL_information["Random_Accuracy"] = random_acc
        # Printing Final Results
        for d in range(len(div_methods)):
            for k in range(len(algorithms)):
                accuracies = AL_information[(AL_information.Algorithm == algorithms[k].__name__) & (AL_information.Diversity_Method == div_methods[d])]
                accuracies = accuracies.iloc[:,[1,4,5]]
                accuracies_avg = accuracies.groupby(["nQueries"]).mean()
                accuracies_avg.columns = ["AL_Acc_Avg", "Random_Acc_Avg"]
                accuracies_std = accuracies.groupby(["nQueries"]).std()
                accuracies_std.columns = ["AL_Acc_Std", "Random_Acc_Std"]
                accuracies = pd.concat([accuracies_avg, accuracies_std], axis=1)
                # Printing Information
                print("Algorithm:", algorithms[k].__name__," | ","Diversity Criterion:", div_methods[d], " | ", "Experiment repeated", num_of_experiments, "times")
                print("---------------------------------------------------------------------------------------")
                for quer in [9, 19, 29, 39]:
                    print(quer+1, "queries:")
                    print("Average Random ACC: %6.3f, Average Active ACC: %6.3f" % (accuracies.iloc[quer,1], accuracies.iloc[quer,0]))
                # Saving positions where non-random algorithm overcomes the random one
                aux = accuracies[accuracies["AL_Acc_Avg"] > accuracies["Random_Acc_Avg"]]
                print("\nTimes", algorithms[k].__name__, "overcomes Random Sampling: ", aux.shape[0])
                # Printing in which n of queries AL algorithm performs better
                if aux.shape[0] == 0:
                    aux1 = accuracies.sort_values(["AL_Acc_Avg"], ascending=False)
                    print("Highest average performance of", algorithms[k].__name__, "with", aux1.index[0], "queries:")
                    aux1 = aux1.reset_index(drop=True)
                    print("Average Random ACC: %6.3f, Average Active ACC: %6.3f" % (aux1.iloc[0,1], aux1.iloc[0,0]))
                else:
                    aux1 = aux.sort_values(["AL_Acc_Avg"], ascending=False)
                    print("Highest average performance of", algorithms[k].__name__, "with", aux1.index[0], "queries:")
                    aux1 = aux1.reset_index(drop=True)
                    print("Average Random ACC: %6.3f, Average Active ACC: %6.3f" % (aux1.iloc[0,1], aux1.iloc[0,0]))
                # Plotting Accuracy Graphs
                x_axis = np.arange(0 , 40)
                plt.figure(figsize=(9, 5))
                plt.suptitle("Average Accuracy Scores for " + str(num_of_experiments) + " iterations" + ", Random Sampling VS " + algorithms[k].__name__, fontsize=16)
                plt.title("Diversity Method: " + div_methods[d], fontsize=12)
                plt.ylabel("Accuracy")
                ax = sns.lineplot(data=accuracies.iloc[:,[0,1]], palette="viridis")
                ax.fill_between(x = x_axis, y1 = np.array(accuracies.AL_Acc_Avg.values + 2*accuracies.AL_Acc_Std.values), y2 = np.array(accuracies.AL_Acc_Avg.values + -(2*accuracies.AL_Acc_Std.values)), alpha = 0.4)
                ax.fill_between(x = x_axis, y1 = np.array(accuracies.Random_Acc_Avg.values + 2*accuracies.Random_Acc_Std.values), y2 = np.array(accuracies.Random_Acc_Avg.values + -(2*accuracies.Random_Acc_Std.values)), alpha = 0.4, color = "#66e882")
                ax.legend(loc="lower right", title = "Algorithms", labels = [algorithms[k].__name__, "Random Sampling"])
                plt.show()
                print("\n")