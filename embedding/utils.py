import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import dataset
import model
import train
from network import Network  # Importing the updated Network class

# Read stId mapping from the graph
def get_stid_mapping(graph):
    stid_mapping = {node_id: data['stId'] for node_id, data in graph.graph_nx.nodes(data=True)}
    return stid_mapping

# Save graph to disk
def save_to_disk(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{graph.kge}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(graph.graph_nx, f)
    print(f"Graph saved to {save_path}")

# Save stId to CSV
def save_stid_to_csv(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stid_data = {'stId': [data['stId'] for _, data in graph.graph_nx.nodes(data=True)]}
    df = pd.DataFrame(stid_data)
    csv_path = os.path.join(save_dir, 'stId_nodes.csv')
    df.to_csv(csv_path, index=False)
    print(f"stId nodes saved to {csv_path}")

# Create network using protein interaction data
def create_network_from_genes(data, kge):
    ##graph = Network('gat/data/split_files/protein_interaction_p_value_results_with_fdr_ptmod.csv') 
    ##graph = Network('gat/data/protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    graph = Network('data/GGNet/ggnet_gene_interaction_p_value_results_with_fdr_two_commons.csv')
    ##graph = Network('gat/data/inhibition_protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    # Initialize the protein network
    graph.interaction_data = data  # Assign the filtered data directly
    graph.build_graph()  # Build the graph with the interaction data
    graph.kge = kge  # Set the knowledge graph embedding identifier
    return graph

# Function to create embedding with genes directly
def create_embedding_with_genes(p_value=0.05, save=True, data_dir='data/emb'):
    # Read protein interaction p-values from CSV
    ##p_value_path = os.path.join('gat/data/split_files/', 'protein_interaction_p_value_results_with_fdr_ptmod.csv')
    ##p_value_path = os.path.join('gat/data/', 'inhibition_protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    ##p_value_path = os.path.join('data/GGNet/', 'ggnet_filtered_methylation.csv')
    p_value_path = os.path.join('data/GGNet/', 'ggnet_gene_interaction_p_value_results_with_fdr_two_commons.csv')
    p_value_df = pd.read_csv(p_value_path)

    # Filter based on p-value threshold
    filtered_df = p_value_df[p_value_df['adjusted_p-value'] <= p_value]
    ##filtered_df = p_value_df[p_value_df['MethylationRatio'] <= p_value]

    # Split the data into train and test sets
    genes_train, genes_test = train_test_split(filtered_df, test_size=0.2, random_state=42)

    # Create networks for train and test sets
    graph_train = create_network_from_genes(genes_train, 'emb_train')
    graph_test = create_network_from_genes(genes_test, 'emb_test')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        os.makedirs(save_dir, exist_ok=True)
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

# Function to create embeddings using GAT model
def create_embeddings(load_model=True, save=True, data_dir='data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and set up directories
    data = dataset.Dataset(data_dir)  # Adjust dataset to handle protein interactions
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    os.makedirs(emb_dir, exist_ok=True)

    # Model parameters
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams.get('num_heads', 2)  # Default to 2 heads if not specified


    dim_latent = hyperparams['out_feats']
    '''num_layers = hyperparams['num_layers']
    
    net = model.SAGEModel(dim_latent=dim_latent, num_layers=num_layers).to(device)'''
    ## net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads).to(device)
    ##net = model.GCNModel(dim_latent=dim_latent, num_layers=num_layers).to(device)
    net = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers).to(device)

    # Load or train the model
    if load_model:
        model_path = os.path.join(data_dir, 'models', 'model.pth')
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    # Generate and save embeddings
    embedding_dict = {}
    for idx in tqdm(range(len(data))):
        graph, name = data[idx]
        graph = graph.to(device)
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding.cpu()

        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)
            print(f"Embedding for {name} saved to {emb_path}")

    return embedding_dict
