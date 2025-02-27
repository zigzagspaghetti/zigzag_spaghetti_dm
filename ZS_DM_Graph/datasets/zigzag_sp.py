import sys
from typing import Optional
import dionysus as d
import zigzag.zigzagtools as zzt
from scipy.stats import multivariate_normal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import networkx as nx
from scipy.spatial.distance import squareform

import math
import dgl
import dgl.function as fn
from utils.utils import make_edge_weights
from .mlp_gat import Denoising_Unet
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos

def single_extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([1] + [1] * (len(x_shape) - 1))

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)

def gaussian_zigzag_filtration_curves(dgm, length):
    weight_i = 1/(length-1)
    initial_mu_i = [0, 0.5]
    sigma_i = [[5, 0], [0, 5]]
    zfc = np.zeros(((length-1)*2, 1))
    for i in range((length-1)*2):
        if i == 0:
            var = multivariate_normal(mean=initial_mu_i, cov=sigma_i)
            zfc[i] = weight_i * np.sum(var.pdf(dgm))
            initial_mu_i = [initial_mu_i[0] + 0.5, initial_mu_i[1] + 0.5]
        else:
            var = multivariate_normal(mean=initial_mu_i, cov=sigma_i)
            zfc[i] = weight_i * np.sum(var.pdf(dgm))
            initial_mu_i = [initial_mu_i[0] + 0.5, initial_mu_i[1] + 0.5]
    return zfc

def zigzag_sp_function(adj, x, noise, scaleParameter, beta_schedule, beta_1, beta_T, T): # adj, x, sorted_t are torch tensor
    betas = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
    sorted_t = torch.tensor(range(T))

    g_adj = (adj).detach().cpu().numpy()
    g_edgelist = []

    for i in range(g_adj.shape[0] - 1):
        for j in range(i, g_adj.shape[0]):
            if g_adj[i, j] == 1:
                g_edgelist.append((int(i), int(j)))

    for i in range(sorted_t.size(0)):
        x_t = (single_extract(sqrt_alphas_bar, sorted_t[i], x.shape) * x + single_extract(
            sqrt_one_minus_alphas_bar, sorted_t[i], x.shape) * noise)
        x_t_np = x_t.detach().cpu().numpy()
        forward_x_ts.append(x_t_np)

    series_networks = np.zeros(shape=(sorted_t.size(0), len(g_edgelist), 3), dtype=np.float32)
    for i in range(sorted_t.size(0)):
        series_networks[i, :, 0:2] = np.array(g_edgelist)
        tmp_features = forward_x_ts[i]
        for j in range(len(g_edgelist)):
            u, v = g_edgelist[j]
            if np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2)) == 0:
                series_networks[i, j, 2] = 1e-5
            else:
                series_networks[i, j, 2] = np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2))

        tmp_max = np.max(series_networks[i, :, 2])
        series_networks[i, :, 2] = series_networks[i, :, 2] / tmp_max

    ##################################
    # Open all sets (point-cloud/Graphs)
    Graphs = []
    for i in range(0, sorted_t.size(0)):
        edgesList = series_networks[i, :, :]
        Graphs.append(edgesList)
    # print("  --- End Loading...")  # Ending

    # Generate Graph
    GraphsNetX = []
    for ii in range(0, sorted_t.size(0)):
        g = nx.Graph()
        g.add_nodes_from(list(range(0, x.size(0))))  # Add vertices...
        if (Graphs[ii].ndim == 1 and len(Graphs[ii]) > 0):
            g.add_edge(int(Graphs[ii][0]), int(Graphs[ii][1]), weight=Graphs[ii][2])
        elif (Graphs[ii].ndim == 2):
            for k in range(0, Graphs[ii].shape[0]):
                g.add_edge(int(Graphs[ii][k, 0]), int(Graphs[ii][k, 1]), weight=Graphs[ii][k, 2])
        GraphsNetX.append(g)

    # Building unions and computing distance matrices
    # print("Building unions and computing distance matrices...")  # Beginning
    GUnions = []
    MDisGUnions = []
    for i in range(0, sorted_t.size(0) - 1):
        # --- To concatenate graphs
        unionAux = []
        MDisAux = np.zeros((2 * x.size(0), 2 * x.size(0)))
        A = nx.adjacency_matrix(GraphsNetX[i]).todense()
        B = nx.adjacency_matrix(GraphsNetX[i + 1]).todense()
        # ----- Version Original (2)
        C = (A + B) / 2
        A[A == 0] = 1.1
        A[range(x.size(0)), range(x.size(0))] = 0
        B[B == 0] = 1.1
        B[range(x.size(0)), range(x.size(0))] = 0
        MDisAux[0:x.size(0), 0:x.size(0)] = A
        C[C == 0] = 1.1
        C[range(x.size(0)), range(x.size(0))] = 0
        MDisAux[x.size(0):(2 * x.size(0)), x.size(0):(2 * x.size(0))] = B
        MDisAux[0:x.size(0), x.size(0):(2 * x.size(0))] = C
        MDisAux[x.size(0):(2 * x.size(0)), 0:x.size(0)] = C.transpose()
        # Distance in condensed form
        pDisAux = squareform(MDisAux)
        # --- To save unions and distances
        GUnions.append(unionAux)  # To save union
        MDisGUnions.append(pDisAux)  # To save distance matrix
    # print("  --- End unions...")  # Ending

    # To perform Ripser computations
    # print("Computing Vietoris-Rips complexes...")  # Beginning

    GVRips = []
    maxDimHoles = 1
    scaleParameter = scaleParameter  # it will be the list
    for jj in range(0, sorted_t.size(0) - 1):
        # print(jj, MDisGUnions[jj])
        ripsAux = d.fill_rips(MDisGUnions[jj], maxDimHoles, scaleParameter)
        GVRips.append(ripsAux)
    # print("  --- End Vietoris-Rips computation")  # Ending

    # Shifting filtrations...
    # print("Shifting filtrations...")  # Beginning
    GVRips_shift = []
    GVRips_shift.append(GVRips[0])  # Shift 0... original rips01
    for kk in range(1, sorted_t.size(0) - 1):
        shiftAux = zzt.shift_filtration(GVRips[kk], sorted_t.size(0) * kk)
        GVRips_shift.append(shiftAux)
    # print("  --- End shifting...")  # Ending

    # To Combine complexes
    # print("Combining complexes...")  # Beginning
    completeGVRips = zzt.complex_union(GVRips[0], GVRips_shift[1])
    for uu in range(2, sorted_t.size(0) - 1):
        completeGVRips = zzt.complex_union(completeGVRips, GVRips_shift[uu])
    # print("  --- End combining")  # Ending

    # To compute the time intervals of simplices
    # print("Determining time intervals...")  # Beginning
    time_intervals = zzt.build_zigzag_times(completeGVRips, sorted_t.size(0), sorted_t.size(0))
    # print("  --- End time")  # Beginning

    # To compute Zigzag persistence
    # print("Computing Zigzag homology...")  # Beginning
    G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeGVRips, time_intervals)
    # print("  --- End Zigzag")  # Beginning

    # To show persistence intervals
    window_ZPD = []
    # Personalized plot
    for vv, dgm in enumerate(G_dgms):
        # print("Dimension:", vv)
        if (vv < 2):
            matBarcode = np.zeros((len(dgm), 2))
            k = 0
            for p in dgm:
                matBarcode[k, 0] = p.birth
                matBarcode[k, 1] = p.death
                k = k + 1
            matBarcode = matBarcode / 2
            window_ZPD.append(matBarcode)

    window_ZPDs = np.concatenate((window_ZPD[0], window_ZPD[1]), axis=0)
    zigzag_out = (gaussian_zigzag_filtration_curves(window_ZPDs, sorted_t.size(0))).transpose()