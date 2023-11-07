## MODULE FOR GENERATING DATASET

# packages
import pandas as pd
import numpy as np
import networkx as nx
from ts2vg import NaturalVG, HorizontalVG
from numpy.linalg import eig
import scipy.stats as stat


def graphIndexComplexity(adjMatrix, graph):
    eigValues, _ = eig(adjMatrix)
    kmax = np.max(eigValues)
    const = 2*np.cos(np.pi/(len(graph)+1))
    C = (kmax - const)/(len(graph) - 1 - const)
    return 4*C*(1-C)

def load_ECG(normalECG_path: str, abnormalECG_path: str, numNormal: int = None, numAbnormal: int = None, return_X_y: bool = False):
    # importing data as pd.DataFrame -> np.ndarray
    DataNormal = pd.read_csv(normalECG_path, header=None)
    DataAbnormal = pd.read_csv(abnormalECG_path, header=None)
    DataNormal = DataNormal.to_numpy()
    DataAbnormal = DataAbnormal.to_numpy()

    if numNormal is None and numAbnormal is None:
        numNormal = len(DataNormal)
        numAbnormal = len(DataAbnormal)
    DataNormal = DataNormal[:numNormal, :]
    DataAbnormal = DataAbnormal[:numAbnormal, :]

    # compiled data
    ECG = dict()
    ECG['timeSeries_data'] = np.append(DataNormal, DataAbnormal, axis=0)
    ECG['target'] = np.append(np.zeros(len(DataNormal)), np.ones(len(DataAbnormal)))
    ECG['target_names'] = np.array(['normal', 'abnormal'])
    ECG['feature_names'] = ['mean_degDist', 'max_degDist', 'min_degDist', 'characteristic_path_length', 'global efficiency', 'ave_clustering_coeff', 'local_eff', 'assortativity_coeff']

    ECG['data'] = []

    for ts in ECG['timeSeries_data']:

        newData = []

        # applying Natural Visibility Graph
        NVG = NaturalVG()
        NVG.build(ts)
        graph = NVG.as_networkx()

        # extract adjacency matrix
        adjMatrix = nx.adjacency_matrix(graph)
        adjMatrix = adjMatrix.toarray()

        # degree distribution
        degDist = [d for _, d in graph.degree()]

        # FEATURE EXTRACTION
        # mean, max, min, std of degDist
        meanDeg = np.mean(degDist)
        maxDeg = np.max(degDist)
        minDeg = np.min(degDist)
        stdDeg = stat.tstd(degDist)
        maxOverMedianDeg = np.max(degDist)/np.median(degDist)

        # characteristic path length
        L = nx.average_shortest_path_length(graph, weight=None)

        # local and global efficiency
        El = nx.local_efficiency(graph)
        Eg = nx.global_efficiency(graph)

        # average clustering coefficient
        C = nx.average_clustering(graph)

        # assortativity coefficient
        r = nx.degree_assortativity_coefficient(graph)

        # additional metrics
        newData.extend([meanDeg, maxDeg, minDeg, stdDeg, L, El, Eg, C, r])

        # final data collection
        ECG['data'].append(newData)
    
    if return_X_y:
        return np.array(ECG['data']), ECG['target']
    else:
        return ECG

def load_ECG_HVG(normalECG_path: str, abnormalECG_path: str, numNormal: int = None, numAbnormal: int = None, return_X_y: bool = False):
    # importing data as pd.DataFrame -> np.ndarray
    DataNormal = pd.read_csv(normalECG_path, header=None)
    DataAbnormal = pd.read_csv(abnormalECG_path, header=None)
    DataNormal = DataNormal.to_numpy()
    DataAbnormal = DataAbnormal.to_numpy()

    if numNormal is None and numAbnormal is None:
        numNormal = len(DataNormal)
        numAbnormal = len(DataAbnormal)
    DataNormal = DataNormal[:numNormal, :]
    DataAbnormal = DataAbnormal[:numAbnormal, :]

    # compiled data
    ECG = dict()
    ECG['timeSeries_data'] = np.append(DataNormal, DataAbnormal, axis=0)
    ECG['target'] = np.append(np.zeros(len(DataNormal)), np.ones(len(DataAbnormal)))
    ECG['target_names'] = np.array(['normal', 'abnormal'])
    ECG['feature_names'] = ['mean_degDist', 'max_degDist', 'min_degDist', 'std_degDist', 'characteristic_path_length', 'local_eff', 'global_eff', 'ave_clustering_coeff', 'assortativity_coeff']
    ECG['data'] = []

    for ts in ECG['timeSeries_data']:

        newData = []

        # applying Horizontal Visibility Graph
        HVG = HorizontalVG()
        HVG.build(ts)
        graph = HVG.as_networkx()

        # extract adjacency matrix
        # adjMatrix = nx.adjacency_matrix(graph)
        # adjMatrix = adjMatrix.toarray()

        # degree distribution
        degDist = [d for _, d in graph.degree()]

        # FEATURE EXTRACTION
        # mean, max, min, std of degDist
        meanDeg = np.mean(degDist)
        maxDeg = np.max(degDist)
        minDeg = np.min(degDist)
        stdDeg = stat.tstd(degDist)

        # characteristic path length
        L = nx.average_shortest_path_length(graph, weight=None)

        # local and global efficiency
        El = nx.local_efficiency(graph)
        Eg = nx.global_efficiency(graph)

        # average clustering coefficient
        C = nx.average_clustering(graph)

        # assortativity coefficient
        r = nx.degree_assortativity_coefficient(graph)

        # additional metrics
        newData.extend([meanDeg, maxDeg, minDeg, stdDeg, L, El, Eg, C, r])

        # final data collection
        ECG['data'].append(newData)
    
    if return_X_y:
        return np.array(ECG['data']), ECG['target']
    else:
        return ECG