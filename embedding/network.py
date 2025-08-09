import pandas as pd
import networkx as nx
from collections import defaultdict

class Network:

    def __init__(self, interaction_data_path):
        # Initialize an empty directed graph
        self.graph_nx = nx.DiGraph()
        
        # Load interaction data from CSV
        self.interaction_data = self.load_interaction_data(interaction_data_path)
        self.build_graph()

    def load_interaction_data(self, path):
        # Load gene interaction data from CSV
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # Strip any leading/trailing whitespace from column names
        return df

    def build_graph(self):
        # Build the graph by adding edges and attributes from the interaction data
        for _, row in self.interaction_data.iterrows():
            Gene1 = row['Gene1']
            Gene2 = row['Gene2']
            stId = row['Gene1']
            name = row['Gene1']
            ##gene_type = row['gene_type']
            shared_partners = row['shared_partners']
            shared_count = row['shared_partners_count']
            ##p_value = row['MethylationRatio']
            p_value = row['adjusted_p-value']
            significance = row['significance']
            
            # Add edge between genes
            self.graph_nx.add_edge(Gene1, Gene2)
            
            # Store gene info in a dictionary format
            self.graph_nx.nodes[Gene1]['stId'] = stId
            self.graph_nx.nodes[Gene1]['name'] = name
            ##self.graph_nx.nodes[Gene1]['gene_type'] = gene_type
            self.graph_nx.nodes[Gene1]['significance'] = significance
            self.graph_nx.nodes[Gene1]['shared_partners'] = shared_partners
            self.graph_nx.nodes[Gene1]['shared_count'] = shared_count
            self.graph_nx.nodes[Gene1]['p_value'] = p_value

    def get_gene_info(self, gene):
        # Retrieve the stored gene info
        if gene in self.graph_nx:
            return {
                'stId': self.graph_nx.nodes[gene]['stId'],
                'name': self.graph_nx.nodes[gene]['name'],
                ##'gene_type': self.graph_nx.nodes[gene]['gene_type'],
                'significance': self.graph_nx.nodes[gene]['significance'],
                'shared_partners': self.graph_nx.nodes[gene]['shared_partners'],
                'shared_count': self.graph_nx.nodes[gene]['shared_count'],
                'p_value': self.graph_nx.nodes[gene]['p_value']
            }
        else:
            return None

    def display_graph(self):
        # Display graph with node attributes (for debugging purposes)
        for node in self.graph_nx.nodes:
            info = self.get_gene_info(node)
            ##print(f"Gene: {node}, StId: {info['stId']}, Name: {info['name']}, Category: {info['gene_type']}, P-value: {info['p_value']}")
            print(f"Gene: {node}, StId: {info['stId']}, Name: {info['name']}, P-value: {info['p_value']}")

    def save_name_to_id(self):
        # Save a mapping of gene names to IDs (stId)
        name_to_id = {node: self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes}
        file_path = 'name_to_id.txt'
        with open(file_path, 'w') as f:
            for name, stid in name_to_id.items():
                f.write(f"{name}: {stid}\n")

    def save_sorted_stids(self):
        # Save a sorted list of gene IDs (stIds)
        file_path = 'sorted_stids.txt'
        stids = sorted([self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes])
        with open(file_path, 'w') as f:
            for stid in stids:
                f.write(f"{stid}\n")

# Usage example
'''interaction_data_path = 'gat/data/gene_interaction_p_value_results_with_fdr.csv'
network = Network(interaction_data_path)
network.display_graph()
network.save_name_to_id()
network.save_sorted_stids()'''
