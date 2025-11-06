import csv
import json
import os

# File paths
##source_csv_path = 'gat/data/ggnet_combined_gene_embeddings_64x2.csv'
##target_csv_path = 'gat/data/ggnet_combined_gene_embeddings_64x2.csv'
source_csv_path = 'gat/data/ggnet_methylation_two_commons_gene_embeddings_lr0.0001_dim128_lay2_epo10_final.csv'
target_csv_path = 'gat/data/ggnet_methylation_two_commons_gene_embeddings_lr0.0001_dim128_lay2_epo10_final.csv'
relation_csv_path = 'gat/data/ggnet_filtered_methylation.csv'
##relation_csv_path = 'gat/data/merged_pathNet_with_gene_type_and_connected_driver_gene.csv'
##relation_csv_path = 'gat/data/ppnet_filtered_expression.csv'
##output_json_path = 'gat/data/ggnet_two_commons_gene_embeddings_lr0.0001_dim128_lay2_epo100_final_64x2.json'
##csv_file_path = 'gat/data/ggnet_two_commons_gene_embeddings_lr0.0001_dim128_lay2_epo100_final_64x2.csv'
output_json_path = 'gat/data/ggnet_methylation_two_commons_gene_embeddings_lr0.0001_dim128_lay2_epo10_final_128x1.json'
csv_file_path = 'gat/data/ggnet_methylation_two_commons_gene_embeddings_lr0.0001_dim128_lay2_epo10_final_128x1.csv'

# Interaction types and corresponding labels
interaction_labels = {
    "0": 0,  # Label 0
    "1": 1,  # Label 1
    "2": -1,  # Label 2
    "3": -1   # Label 3
}

# Ensure interaction stats CSV exists with a header
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['num_nodes', 'num_edges']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

# Function to read embeddings from a CSV file
def read_embeddings(file_path):
    embeddings = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header
        for row in reader:
            name = row[0]
            embedding = list(map(float, row[1:]))
            embeddings[name] = embedding
    return embeddings

# Read source and target embeddings
source_embeddings = read_embeddings(source_csv_path)
target_embeddings = read_embeddings(target_csv_path)

# Read relationships and count nodes and edges
nodes = set()
edges = 0
relationships_to_include = []

with open(relation_csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        source_stId = row['Gene1']
        target_stId = row['Gene2']
        relation_type = row['gene_type']
        if relation_type in interaction_labels:
            nodes.add(source_stId)
            nodes.add(target_stId)
            relationships_to_include.append((source_stId, target_stId, relation_type))
            edges += 1

# Print node and edge counts
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {edges}")
# Create the JSON structure
relationships = []
for source_stId, target_stId, relation_type in relationships_to_include:
    if source_stId in source_embeddings and target_stId in target_embeddings:
        source_label = interaction_labels[relation_type]  # Assign label based on type

        # Assign label 1 to target node if gene_type (relation_type) is 2
        ##target_label = 1 if relation_type == "2" else None
        
        target_label = 1 if relation_type == "2" else 0 if relation_type == "0" else None

        relationship = {
            "source": {
                "properties": {
                    "name": source_stId,
                    "label": source_label,  # Add label based on gene type
                    "embedding": source_embeddings[source_stId]
                }
            },
            "relation": {
                "type": relation_type  # Dynamically set relation type
            },
            "target": {
                "properties": {
                    "name": target_stId,
                    "label": target_label,  # Add label to target node
                    "embedding": target_embeddings[target_stId]
                }
            }
        }
        relationships.append(relationship)

# Save to JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(relationships, json_file, indent=2)

print(f"JSON file saved to {output_json_path}")

# Save node and edge counts to CSV file
with open(csv_file_path, 'a', newline='') as csvfile:
    fieldnames = ['num_nodes', 'num_edges']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({
        'num_nodes': len(nodes),
        'num_edges': edges
    })

print(f"CSV file updated with node and edge counts.")

'''import csv
import json
import os
from collections import defaultdict

# File paths
input_json_path = 'gat/data/ppnet_combined_gene_embeddings_32_64_label_0_1_2_3.json'
output_csv_path = 'gat/results/gene_prediction/nodes_label_2_degree.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Load the JSON graph
with open(input_json_path, 'r') as json_file:
    relationships = json.load(json_file)

# Initialize degree counter
degree_counts = defaultdict(int)

# Process relationships
for relationship in relationships:
    source = relationship["source"]["properties"]
    target = relationship["target"]["properties"]

    # Check for label 2 (source) connecting to label 1 (target)
    if source.get("label") == 2 and target.get("label") == 1:
        degree_counts[source["name"]] += 1

# Save the degrees to a CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Node", "Degree"])  # Header row
    for node, degree in degree_counts.items():
        writer.writerow([node, degree])

print(f"Degrees of nodes labeled as 2 connected to nodes labeled as 1 saved to: {output_csv_path}")
'''