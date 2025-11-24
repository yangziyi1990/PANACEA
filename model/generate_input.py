import glob
from collections import Counter
import pandas as pd
import numpy as np
import random
import networkx as nx
import torch
from torch_geometric.data import Data


def split_data(num_y):
    split_idx = list(range(num_y))
    random.shuffle(split_idx)
    train_idx = split_idx[ : int(len(split_idx) * 0.8)] # Train mask
    train_mask = torch.zeros(num_y, dtype = torch.bool)
    train_mask[train_idx] = 1
    val_idx = split_idx[ int(len(split_idx) * 0.8) : int(len(split_idx) * 0.9)] # Val mask
    val_mask = torch.zeros(num_y, dtype=torch.bool)
    val_mask[val_idx] = 1
    test_idx = split_idx[int(len(split_idx) * 0.9) : ] # Test mask
    test_mask = torch.zeros(num_y, dtype=torch.bool)
    test_mask[test_idx] = 1
    return train_mask, val_mask, test_mask


def read_ppi(ppi_dir):
    orig_ppi_layers = dict()
    ppi_layers = dict()
    ppi_train = dict()
    ppi_val = dict()
    ppi_test = dict()
    
    for f in glob.glob(ppi_dir + "*.txt"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>

        # Parse name of context （GO name）
        context_parts = f.split(ppi_dir)[1].split(".")[0].split("_")
        if len(context_parts) >= 4 and context_parts[0] == "GO":
            go_id = context_parts[1]
            go_aspect = context_parts[2]
            go_name = " ".join(context_parts[3:])
            context = f"GO:{go_id}_{go_aspect}:{go_name}"
        else:
            # Fallback to previous behaviour when the filename does not match the
            # expected GO naming pattern.
            context = f.split(ppi_dir)[1].split(".")[0].replace("_", " ")

        # Read edgelist
        ppi = nx.read_edgelist(f)

        # Relabel PPI nodes
        mapping = {n: idx for idx, n in enumerate(ppi.nodes())}
        ppi_layers[context] = nx.relabel_nodes(ppi, mapping)
        orig_ppi_layers[context] = ppi
        
        # 这里要求图必需是联通的，我们的数据现在不满足这个情况，可以注释掉
        # assert nx.is_connected(ppi_layers[context])

        # Split into train/val/test
        ppi_train[context], ppi_val[context], ppi_test[context] = split_data(len(ppi_layers[context].edges))
    return orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test


def create_data(G, train_mask, val_mask, test_mask, node_type, edge_type, x):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    y = torch.ones(edge_index.size(1))
    num_classes = len(torch.unique(y))
    node_type = torch.tensor(node_type)
    edge_type = torch.tensor(edge_type)
    new_G = Data(x = x, y = y, num_classes = num_classes, edge_index = edge_index, node_type = node_type, edge_attr = edge_type, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    return new_G


def read_global_ppi(f):
    # Read table from csv file
    graph_df = pd.read_csv(f)

    # Create a list of tuples, where each tuple is an edge
    edges = [(s, t) for s, t in zip(graph_df["protein1"].tolist(), graph_df["protein2"].tolist())]

    # Instantiate graph object
    G = nx.Graph()

    # Add edges (from the table) to the graph object
    G.add_edges_from(edges)

    return G


def read_data(G_f, ppi_dir, mg_f, feat_mat_dim):

    # Read global PPI 
    G = nx.read_edgelist(G_f)
    #G = read_global_ppi(G_f)

    feat_mat = torch.normal(torch.zeros(len(G.nodes), feat_mat_dim), std=1)
    
    # Read PPI layers
    orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test = read_ppi(ppi_dir)
    print("Number of PPI layers:", len(ppi_layers), len(ppi_train), len(ppi_val), len(ppi_test))

    # Read metagraph （有向图）
    metagraph = nx.read_edgelist(mg_f, data=False, delimiter = "\t", create_using=nx.DiGraph)
    
    # 要求 meta-gtaph 是联通的
    #assert nx.is_connected(metagraph.to_undirected())
    
    mg_feat_mat = torch.zeros(len(metagraph.nodes), feat_mat_dim)
    
    orig_mg = metagraph
    print("Number of nodes:", len(metagraph.nodes), "Number of edges:", len(metagraph.edges))
    print(ppi_layers)
    # Map every node in the metagraph to an index. Some GO contexts may not have a
    # corresponding PPI layer, so build the mapping from the metagraph itself to
    # avoid missing nodes.
    mg_mapping = {n: i for i, n in enumerate(sorted(metagraph.nodes))}

    # Warn when the metagraph contains nodes that do not have PPI layers. The
    # subsequent filtering of PPI-related structures uses this mapping, so the
    # pipeline can continue without raising an assertion.
    missing_ppi_layers = set(metagraph.nodes) - set(ppi_layers)
    if missing_ppi_layers:
        print("Metagraph nodes without PPI layers detected:", missing_ppi_layers)
    #print(mg_mapping)

    # Set up Data object
    mg_nodetype = [0 if "STRAIN" in n else 1 for n in mg_mapping] # Strains nodes = 0, GO-type nodes = 1, gene nodes = 2
    mg_edgetype = []
    for edges in metagraph.edges:
        if "STRAIN" in edges[0] and "STRAIN" in edges[1]: mg_edgetype.append(0) # strain-strain edge
        elif "STRAIN" in edges[0] and "STRAIN" not in edges[1]: mg_edgetype.append(1) # strain-GO edge
        elif "STRAIN" not in edges[0] and "STRAIN" in edges[1]: mg_edgetype.append(2) # GO-strain edge
        elif "STRAIN" not in edges[0] and "STRAIN" not in edges[1]: mg_edgetype.append(3) # GO-GO edge
        else:
            print(edges)
            raise NotImplementedError
    tissue_neighbors = {mg_mapping[t]: [mg_mapping[n] for n in metagraph.to_undirected().neighbors(t)] for t in metagraph.to_undirected().nodes if "STRAIN" in t}
    metagraph = nx.relabel_nodes(metagraph, mg_mapping)
    mg_mask = torch.ones(len(metagraph.edges), dtype = torch.bool) # Pass in all meta graph edges during training, validation, and test
    mg_data = create_data(metagraph, mg_mask, mg_mask, mg_mask, mg_nodetype, mg_edgetype, mg_feat_mat)

    # Set up PPI Data objects
    orig_ppi_layers_remapped = {mg_mapping[k]: v for k, v in orig_ppi_layers.items() if k in mg_mapping}
    ppi_layers = {mg_mapping[k]: v for k, v in ppi_layers.items() if k in mg_mapping}
    ppi_train = {mg_mapping[k]: v for k, v in ppi_train.items() if k in mg_mapping}
    ppi_val = {mg_mapping[k]: v for k, v in ppi_val.items() if k in mg_mapping}
    ppi_test = {mg_mapping[k]: v for k, v in ppi_test.items() if k in mg_mapping}    
    ppi_data = dict()
    for cluster, ppi in ppi_layers.items():
        ppi_nodetype = [2] * len(ppi.nodes) # protein nodes = 2
        ppi_edgetype = [4] * len(ppi.edges) # protein-protein edge
        ppi_node_names = orig_ppi_layers_remapped[cluster].nodes()
        p_index = [idx for idx, p in enumerate(list(G.nodes())) if p in ppi_node_names]
        assert len(p_index) == len(ppi_node_names)
        c_feat_mat = feat_mat[p_index, :]
        assert c_feat_mat.shape[0] == len(ppi.nodes)
        ppi_data[cluster] = create_data(ppi, ppi_train[cluster], ppi_val[cluster], ppi_test[cluster], ppi_nodetype, ppi_edgetype, c_feat_mat)

    #  Set up edge attr dict
    edge_attr_dict = {"tissue_tissue": 0, "tissue_cell": 1, "cell_tissue": 2, "cell_cell": 3, "protein_protein": 4}
    
    # Return celltype specific PPI network data
    return ppi_data, mg_data, edge_attr_dict, mg_mapping, tissue_neighbors, orig_ppi_layers, orig_mg


def subset_ppi(num_subset, ppi_data, ppi_layers):
    
    # Take a subset of PPI data objects
    new_ppi_data = dict()
    for celltype, ppi in ppi_data.items():
        if len(new_ppi_data) < num_subset:
            new_ppi_data[celltype] = ppi
    ppi_data = new_ppi_data

    # Take a subset of PPI layers
    new_ppi_layers = dict()
    for celltype, ppi in ppi_layers.items():
        if len(new_ppi_layers) < num_subset:
            new_ppi_layers[celltype] = ppi_layers[celltype]
    ppi_layers = new_ppi_layers

    return ppi_data, ppi_layers


def get_metapaths():
    ppi_metapaths = [[4]] # Get PPI metapaths
    mg_metapaths = [[0], [1], [2], [3]]
    return ppi_metapaths, mg_metapaths


def get_centerloss_labels(args, celltype_map, ppi_layers):
    center_loss_labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    print(celltype_map)
    reverse_celltype_map = {v: k for k, v in celltype_map.items()}
    for celltype, ppi in ppi_layers.items():
        if celltype in celltype_map:
            label = celltype_map[celltype]
        elif celltype in reverse_celltype_map:
            label = celltype
        else:
            raise KeyError(f"Celltype {celltype} missing from celltype_map")
        center_loss_labels += [label] * len(ppi.nodes)
    center_loss_idx = random.sample(range(len(center_loss_labels)), len(center_loss_labels))
    train_mask = center_loss_idx[ : int(0.8 * len(center_loss_idx))]
    val_mask = center_loss_idx[len(train_mask) : len(train_mask) + int(0.1 * len(center_loss_idx))]
    test_mask = center_loss_idx[len(train_mask) + len(val_mask) : ]
    print("Center loss labels:", Counter(center_loss_labels))
    return center_loss_labels, train_mask, val_mask, test_mask
