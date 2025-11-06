import os
import pickle
import torch
import dgl
import utils
import model
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create embeddings and save to disk.')

    # ===== Arguments =====
    parser.add_argument('--data_dir', type=str, default='data/emb',
                        help='Directory to save the data.')

    parser.add_argument('--output-file', type=str, default='data/emb/embeddings.pkl',
                        help='File to save the embeddings')

    parser.add_argument('--p_value', type=float, default=0.05,
                        help='P-value threshold.')

    parser.add_argument('--save', type=bool, default=True,
                        help='Flag to save embeddings.')

    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='Number of epochs.')

    parser.add_argument('--in_feats', type=int, default=20,
                        help='Input feature dim.')

    parser.add_argument('--out_feats', type=int, default=128,
                        help='Latent output dim.')

    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GNN layers.')

    parser.add_argument('--num_heads', type=int, default=1,
                        help='GAT heads.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')

    parser.add_argument('--print-embeddings', action='store_true',
                        help='Print embedding dict.')

    # UPDATED: added TAGCN & consistent model list
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['TAGCN', 'GraphSAGE', 'GAT', 'GCN', 'GIN', 'Chebnet'],
        required=True,
        help="Which GNN model to use."
    )

    args = parser.parse_args()
    # utils.create_embedding_with_genes(
    #     ##p_value=args.p_value, 
    #     save=args.save, 
    #     data_dir=args.data_dir
    # )
    
    # ================ Collect hyperparameters ================
    hyperparameters = {
        'model_type': args.model_type,
        'num_epochs': args.num_epochs,
        'in_feats': args.in_feats,
        'out_feats': args.out_feats,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'batch_size': args.batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': args.lr,
    }

    # ================ Create embeddings ================
    embedding_dict = utils.create_embeddings(
        data_dir=args.data_dir,
        load_model=False,
        hyperparams=hyperparameters
    )

    if args.print_embeddings:
        print(embedding_dict)

    # ================ Save to pickle ================
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(embedding_dict, f)

    print(f"Embeddings saved to {args.output_file}")


if __name__ == '__main__':
    main()
