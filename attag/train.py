## python _gene_label_prediction_tsne_pertag.py --model_type ATTAG --net_type pathnet --score_threshold 0.4 --learning_rate 0.001 --num_epochs 65 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python _gene_label_prediction_tsne_sage.py --model_type EMOGI --net_type ppnet --score_threshold 0.5 --learning_rate 0.001 --num_epochs 100 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python _gene_label_prediction_tsne_pertag.py --model_type ATTAG --net_type ppnet --score_threshold 0.9 --learning_rate 0.001 --num_epochs 201
import argparse
import json
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn import SAGEConv, GATConv, GraphConv
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from collections import defaultdict
import scipy.stats
from scipy.stats import spearmanr
import pandas as pd
from dgl.nn.pytorch import GINConv
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch import TAGConv
from tqdm import tqdm
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
import numpy as np
from models import ATTAG, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, Chebnet, FocalLoss
from utils import (choose_model, plot_roc_curve, plot_pr_curve, load_graph_data, 
                       load_oncokb_genes, plot_and_analyze, save_and_plot_results)


def train(args):
    # Load data
    ##data_path = os.path.join('data/', f'{args.net_type}_omics_filtered_combined_gene_embeddings_32x4.json')
    data_path = os.path.join('data/', f'{args.net_type}_omics_filtered_combined_gene_embeddings_128x1.json')
    print('data_path=============================', data_path)
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    
    graph = dgl.add_self_loop(graph)
    # Hyperparameters
    in_feats = embeddings.shape[1]
    hidden_feats = args.hidden_feats
    out_feats = 1  # Binary classification (driver gene or not)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = choose_model(args.model_type, in_feats, hidden_feats, out_feats)
    model = model.to(args.device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Move data to device
    graph = graph.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device).float()  # BCE loss expects float labels
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)
    
    # Training loop
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}")

    # Testing and predictions
    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
        print("Predicted Scores:", scores[test_mask.cpu().numpy()])

    # Rank nodes by scores for only non-labeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)
    
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    drivers_file_path = "data/796_drivers.txt"
    oncokb_file_path = "data/oncokb_1172.txt"
    ongene_file_path = "data/ongene_803.txt"
    ncg_file_path = "data/ncg_8886.txt"
    intogen_file_path = "data/intogen_23444.txt"

    # Load data from the confirmation files
    with open(oncokb_file_path, 'r') as f:
        oncokb_genes = set(line.strip() for line in f)

    with open(ongene_file_path, 'r') as f:
        ongene_genes = set(line.strip() for line in f)

    with open(ncg_file_path, 'r') as f:
        ncg_genes = set(line.strip() for line in f)

    with open(intogen_file_path, 'r') as f:
        intogen_genes = set(line.strip() for line in f)

    # Threshold for the score
    score_threshold = args.score_threshold
    confirmed_predictions = []
    predicted_genes = []  # Store predictions with or without confirmation
    for node, score in ranking:
        if score >= score_threshold:
            sources = []  # Accumulate sources confirming the gene
            if node in oncokb_genes:
                sources.append("OncoKB")
            if node in ongene_genes:
                sources.append("OnGene")
            if node in ncg_genes:
                sources.append("NCG")
            if node in intogen_genes:
                sources.append("IntOGen")
            if sources:  # If the gene is confirmed by at least one source
                confirmed_predictions.append((node, score, ", ".join(sources)))
            # Add prediction with sources if confirmed, or 'None' if no confirmation
            predicted_genes.append((node, score, ", ".join(sources) if sources else ""))

    # Save predictions to a CSV file
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    predicted_genes_csv_path = os.path.join(output_dir, f'{args.net_type}_{args.model_type}_predicted_driver_genes_with_confirmed_sources_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv')
    df_predictions = pd.DataFrame(predicted_genes, columns=["Gene", "Score", "Confirmed Sources"])
    df_predictions.to_csv(predicted_genes_csv_path, index=False)

    print(f"Predicted driver genes with confirmed sources saved to {predicted_genes_csv_path}")
    confirmed_predictions_csv_path = os.path.join(output_dir, f'{args.net_type}_{args.model_type}_confirmed_predicted_genes_with_sources_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv')
    df_confirmed = pd.DataFrame(confirmed_predictions, columns=["Gene", "Score", "Source"])
    df_confirmed.to_csv(confirmed_predictions_csv_path, index=False)

    print(f"Confirmed predicted genes saved to {confirmed_predictions_csv_path}")

    # Load known cancer driver genes from the file
    with open(drivers_file_path, 'r') as f:
        known_drivers = set(line.strip() for line in f)

    # Collect predicted cancer driver genes that match the known drivers
    predicted_driver_genes = [node_names[i] for i in non_labeled_nodes if node_names[i] in known_drivers]

    # Save the predicted known cancer driver genes to a CSV file
    predicted_drivers_csv_path = os.path.join(output_dir, f'{args.net_type}_{args.model_type}_predicted_known_drivers_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv')
    df = pd.DataFrame(predicted_driver_genes, columns=["Gene"])
    df.to_csv(predicted_drivers_csv_path, index=False)
    print(f"Predicted known driver genes saved to {predicted_drivers_csv_path}")
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]
    predicted_driver_nodes_above_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] >= args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )
    predicted_driver_nodes_below_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] < args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )

    from collections import defaultdict
    import statistics

    ground_truth_driver_indices = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_driver_names = {node_names[i] for i in ground_truth_driver_indices}
    output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_predicted_driver_genes_above_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv'
    )
    output_file_below = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_predicted_driver_genes_below_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv'
    )

    with open(output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_above_threshold)

    with open(output_file_below, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_below_threshold)
    degree_counts_above = defaultdict(int)
    degree_counts_below = defaultdict(int)

    for src, dst in edges:
        src_name = node_names[src]
        dst_name = node_names[dst]

        # Count only connections to ground truth driver genes (label 1 nodes)
        if dst_name in ground_truth_driver_names:
            if src_name in [gene for gene, _ in predicted_driver_nodes_above_threshold]:
                degree_counts_above[src_name] += 1
            elif src_name in [gene for gene, _ in predicted_driver_nodes_below_threshold]:
                degree_counts_below[src_name] += 1

    # Sort degrees by degree count in descending order
    sorted_degree_counts_above = sorted(degree_counts_above.items(), key=lambda x: x[1], reverse=True)
    sorted_degree_counts_below = sorted(degree_counts_below.items(), key=lambda x: x[1], reverse=True)

    # Save degrees of predicted driver genes connecting to ground truth driver genes (above threshold)
    degree_output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_predicted_driver_gene_degrees_above_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv'
    )
    with open(degree_output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Predicted Driver Gene', 'Degree'])  # Header row
        csvwriter.writerows(sorted_degree_counts_above)
        ##csvwriter.writerow(['Average Degree', average_degree_above])  # Save average degree

    print(f"Degrees of predicted driver genes (above threshold) saved to {degree_output_file_above}")

    # Save degrees of nodes with scores below the threshold (connecting only to label 1 nodes)
    degree_output_file_below = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_predicted_driver_gene_degrees_below_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.csv'
    )
    with open(degree_output_file_below, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Node', 'Degree'])  # Header row
        csvwriter.writerows(sorted_degree_counts_below)
        ##csvwriter.writerow(['Average Degree', average_degree_below])  # Save average degree

    print(f"Degrees of nodes with scores below threshold saved to {degree_output_file_below}")

    nodes_with_degrees = []

    # Above threshold
    for gene, degree in degree_counts_above.items():
        nodes_with_degrees.append({'Gene_Set': 'Above Threshold', 'Degree': degree})

    # Below threshold
    for gene, degree in degree_counts_below.items():
        nodes_with_degrees.append({'Gene_Set': 'Below Threshold', 'Degree': degree})

    # Convert to DataFrame
    nodes_with_degrees = pd.DataFrame(nodes_with_degrees)

    ##print(f"sorted_degree_counts_below: {sorted_degree_counts_below}")
    sorted_degree_counts_above_value = [value for _, value in sorted_degree_counts_above if value <= 30]
    sorted_degree_counts_below_value = [value for _, value in sorted_degree_counts_below if value <= 30]

    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]

    predicted_driver_nodes = [node_names[i] for i in non_labeled_nodes if scores[i] >= args.score_threshold]

    # Prepare data to save to CSV
    stats_output_file = os.path.join('results/gene_prediction/', f'{args.net_type}_{args.model_type}_prediction_stats_{args.num_epochs}.csv')
    with open(stats_output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Non-Labeled Nodes Count', 'Driver Genes', 'Non-Driver Genes', 'Total Testing Nodes', 'Predicted Driver Genes'])
        csvwriter.writerow([
            non_labeled_nodes_count,
            len(ground_truth_driver_nodes),
            len(ground_truth_non_driver_nodes),
            len(ground_truth_driver_nodes) + len(ground_truth_non_driver_nodes),
            len(predicted_driver_nodes)
        ])

    print(f"Prediction statistics saved to {stats_output_file}")

    degree_data = [sorted_degree_counts_above_value, sorted_degree_counts_below_value]

    # Create the box plot
    plt.figure(figsize=(3, 4))

    # Create the boxplot
    boxplot = plt.boxplot(
        degree_data,
        vert=True,
        patch_artist=True,  # Allows customization of box color
        flierprops=dict(marker='o', markerfacecolor='grey', markeredgecolor='grey', markersize=5, alpha=0.2),  # Half-transparent gray dots for outliers
        boxprops=dict(facecolor='green', color='black'),  # Color for boxes
        medianprops=dict(color='blue', linewidth=2),  # Style for median lines
        whiskerprops=dict(color='black', linewidth=1.5),  # Style for whiskers
        capprops=dict(color='black', linewidth=1.5)  # Style for caps
    )

    # Customize frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove the top frame line
    ax.spines['right'].set_visible(False)  # Remove the right frame line

    # Add labels and title
    plt.xticks([1, 2], ['PCGs', 'Other'], fontsize=8)  # Category labels
    plt.yticks(fontsize=8)
    plt.ylabel('Interaction degrees with KCGs', fontsize=10, labelpad=10) 

    # Add different bar colors for each category
    colors = ['green', 'skyblue']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Save the plot
    output_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_degree_distributions_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.png'
    )
    plt.savefig(output_plot_path, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()

    print(f"Box plot saved to {output_plot_path}")

    print("Preparing data for KDE plot...")
    scores = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
    degrees = [
        degree_counts_above.get(node_names[i], 0) +
        degree_counts_below.get(node_names[i], 0)
        for i in range(len(node_names))
    ]

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "Prob_pos_ranked": pd.Series(scores).rank(),
        "Degree_ranked": pd.Series(degrees).rank()
    })

    # Plot KDE
    print("Generating KDE plot...")
    plt.figure(figsize=(4, 4))
    kde_plot = sns.kdeplot(
        x=plot_data["Prob_pos_ranked"],
        y=plot_data["Degree_ranked"],
        cmap="Reds",  # Gradient colormap
        ##shade=True,
        fill=True,
        alpha=0.7,  # Transparency
        levels=50,  # Contour levels
        thresh=0.05  # Threshold to filter low-density regions
    )

    # Calculate Spearman correlation
    correlation, p_value = scipy.stats.spearmanr(
        plot_data["Prob_pos_ranked"],
        plot_data["Degree_ranked"]
    )

    # Add labels and titles
    plt.xticks(fontsize=8)  # Category labels
    plt.yticks(fontsize=8)
    plt.xlabel('PEGNN score rank', fontsize=10, labelpad=10)  # Adjusted label distance
    plt.ylabel('KCG interaction rank', fontsize=12, labelpad=10)
    plt.gca().tick_params(axis='both', labelsize=8)

    # Manually add legend
    legend_text = f"Spearman R: {correlation:.4f}\nP-value: {p_value:.3e}"
    plt.text(
        0.05, 0.95, legend_text,
        fontsize=8, transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    # Save the KDE plot
    kde_output_path = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_kde_plot_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.png'
    )
    plt.savefig(kde_output_path, bbox_inches='tight')
    print(f"KDE plot saved to {kde_output_path}")

    # Show plot
    plt.tight_layout()
    plt.show()

    labeled_scores = scores[train_mask.cpu().numpy()] if isinstance(scores, torch.Tensor) else scores[train_mask.cpu().numpy()]
    labeled_labels = labels[train_mask.cpu().numpy()] if isinstance(labels, torch.Tensor) else labels[train_mask.cpu().numpy()]

    output_file_roc = os.path.join('results/gene_prediction/', f'{args.net_type}_{args.model_type}_threshold{args.score_threshold}_epo{args.num_epochs}_128x1_roc_curves.png')
    output_file_pr = os.path.join('results/gene_prediction/', f'{args.net_type}_{args.model_type}_threshold{args.score_threshold}_epo{args.num_epochs}_128x1_pr_curves.png')

    # Convert labeled_labels and labeled_scores to NumPy arrays if they are PyTorch tensors
    if isinstance(labeled_scores, torch.Tensor):
        labeled_scores_np = labeled_scores.cpu().detach().numpy()
    else:
        labeled_scores_np = labeled_scores

    if isinstance(labeled_labels, torch.Tensor):
        labeled_labels_np = labeled_labels.cpu().detach().numpy()
    else:
        labeled_labels_np = labeled_labels

    # Plot curves
    plot_roc_curve(labeled_labels_np, labeled_scores_np, output_file_roc)
    plot_pr_curve(labeled_labels_np, labeled_scores_np, output_file_pr)

    # Define models and networks
    models = ["ATTAG", "GAT", "HGDC", "EMOGI", "MTGCN", "GCN", "Chebnet", "GraphSAGE", "GIN"]
    networks = ["Protein Network", "Pathway Network", "Gene Network"]

    # AUPRC values for ONGene and OncoKB for each model (rows: models, cols: networks)
    auroc = [
        [0.8285, 0.9647, 0.9723],  # ATTAG
        [0.7689, 0.9190, 0.7021],  # GAT
        [0.7471, 0.9167, 0.7078],  # HGDC
        [0.6885, 0.9196, 0.7358],  # EMOGI
        [0.7199, 0.7932, 0.7664],  # MTGCN
        [0.7254, 0.8317, 0.7681],  # GCN
        [0.8636, 0.9539, 0.8686],  # Chebnet
        [0.8338, 0.9747, 0.9403],  # GraphSAGE
        [0.5854, 0.9193, 0.9293]   # GIN
    ]

    auprc = [
        [0.9700, 0.9748, 0.9854],  # ATTAG
        [0.9452, 0.9430, 0.8066],  # GAT
        [0.9408, 0.9343, 0.7999],  # HGDC
        [0.9251, 0.9432, 0.8260],  # EMOGI
        [0.9122, 0.8392, 0.8575],  # MTGCN
        [0.9329, 0.8829, 0.8579],  # GCN
        [0.9760, 0.9687, 0.9217],  # Chebnet
        [0.9703, 0.9533, 0.9659],  # GraphSAGE
        [0.8941, 0.9346, 0.9611]   # GIN
    ]

    # Compute averages for each model
    average_ongene = np.mean(auroc, axis=1)
    average_oncokb = np.mean(auprc, axis=1)

    # Define colors for models and unique shapes for networks
    colors = ['red', 'grey', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']
    network_markers = ['P', '^', 's']  # One shape for each network
    ##markers = ['o', 's', 'D', '^', 'P', '*']
    average_marker = 'o'

    # Plotting
    plt.figure(figsize=(8, 7))

    # Plot individual points for each model and network
    for i, model in enumerate(models):
        for j, network in enumerate(networks):
            plt.scatter(auprc[i][j], auroc[i][j], color=colors[i], 
                        marker=network_markers[j], s=90, alpha=0.6)

    # Add average points for each model
    for i, model in enumerate(models):
        plt.scatter(average_oncokb[i], average_ongene[i], color=colors[i], marker=average_marker, 
                    s=240, edgecolor='none', alpha=0.5)

    # Create legends for models (colors) and networks (shapes)
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                            markersize=14, label=models[i], alpha=0.5) for i in range(len(models))]
    network_legend = [Line2D([0], [0], marker=network_markers[i], color='k', linestyle='None', 
                            markersize=8, label=networks[i]) for i in range(len(networks))]

    # Add network legend (bottom-right, unique shapes)
    network_legend_artist = plt.legend(handles=network_legend, loc='lower right', title="Networks", fontsize=12, title_fontsize=14, frameon=True)
    plt.gca().add_artist(network_legend_artist)

    # Add model legend (top-left, colors)
    ##plt.legend(handles=model_legend, loc='upper left', title=" ", fontsize=10, title_fontsize=12, frameon=True)
    plt.legend(handles=model_legend, loc='upper left', fontsize=12, frameon=True)


    # Labels and title
    plt.ylabel("AUPRC", fontsize=14)
    plt.xlabel("AUROC", fontsize=14)
    ##plt.title("Comparison of Models and Networks", fontsize=14)

    comp_output_path = os.path.join(
        'results/gene_prediction/',
        f'{args.net_type}_{args.model_type}_comp_plot_threshold{args.score_threshold}_epo{args.num_epochs}_128x1.png'
    )
    plt.savefig(comp_output_path, bbox_inches='tight')
    print(f"KDE plot saved to {kde_output_path}")
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-based driver gene prediction")
    ##parser.add_argument('--data_path', type=str, default='data/pathnet_combined_gene_embeddings_32_64_label_0_1_2_3_bidirection.json', help="Path to the input JSON data file")
    ##parser.add_argument('--output_file', type=str, default='results/gene_prediction/predicted_driver_genes.csv', help="Path to save the predicted rankings")
    parser.add_argument('--in_feats', type=int, default=128, help="Number of in features in GNN layers")
    parser.add_argument('--hidden_feats', type=int, default=128, help="Number of hidden features in GNN layers")
    parser.add_argument('--out_feats', type=int, default=1, help="Number of out features in GNN layers")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--model_type', type=str, choices=['GraphSAGE', 'GAT', 'HGDC', 'EMOGI', 'MTGCN', 'GCN', 'GIN', 'Chebnet', 'ATTAG'], required=True, help="Type of GNN model to use")
    parser.add_argument('--net_type', type=str, choices=['pathnet', 'ppnet', 'ggnet'], required=True, help="Type of gene net to use")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on")
    parser.add_argument('--score_threshold', type=float, default=0.85, help="Score threshold for identifying predicted driver genes")

    args = parser.parse_args()

    train(args)
    ##plot_and_analyze(args)

