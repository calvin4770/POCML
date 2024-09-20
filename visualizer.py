from sklearn.manifold import * 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb

import networkx as nx

def batch_visualize(distances, legend:str ="Node", methods = "mds", log = False):
    """
    Visualize pairwise distances using various dimensionality reduction methods (e.g., MDS, t-SNE).

    Args:
        distances: A matrix of pairwise distances.
        legend (str): The label to use for legends in the visualization (default: "Node").
        methods (str or list or dict): Specifies which (methods : arguments) to use for visualization.
                              - It can be a single string (e.g., "mds"), or a list of method names; arguments are padded empty up to constraints from the methods.
                              - If set to "all", it will automatically include all implemented methods.
        log (bool): Whether to log the process.
    """
    mtds = methods

    if isinstance(mtds, str):
        mtds = [mtds]
    
    if isinstance(mtds, list):
        mtds = {item: {} for item in mtds}

    if "all" in mtds:
        for mtd in ["mds", "tsne", "isomap", "lle", "se"]:
            if mtd not in mtds: 
                mtds[mtd] = {}

    print(mtds)

    if ("mds" in mtds): 
        ml = MDS(**mtds["mds"])
        visualize(distances, ml, legend = legend, method_name = "MDS", log = log)

    if ("tsne" in mtds): 
        if "perplexity" not in mtds["tsne"]:
            mtds["tsne"]["perplexity"] = min(len(distances)-1, 5)          # TODO: recommend #actions for state visualization
        
        ml = TSNE(**mtds["tsne"])
        visualize(distances, ml, legend = legend, method_name = "tSNE", log = log)

    # Isomap
    if "isomap" in mtds:

        if "n_neighbors" not in mtds["isomap"]:
            mtds["isomap"]["n_neighbors"] = min(len(distances)-1, 5)          # TODO

        ml = Isomap(**mtds["isomap"])
        visualize(distances, ml, legend=legend, method_name="Isomap", log=log)

    # Locally Linear Embedding (LLE)
    if "lle" in mtds:

        if "n_neighbors" not in mtds["lle"]:
            mtds["lle"]["n_neighbors"] = min(len(distances)-1, 5)          # TODO
    
        ml = LocallyLinearEmbedding(**mtds["lle"])
        visualize(distances, ml, legend=legend, method_name="LLE", log=log)

    # Spectral Embedding (SE)
    if "se" in mtds:
        # if "n_components" not in mtds["se"]:
        #     mtds["se"]["n_neighbors"] = min(len(distances)-1, 2)          # TODO
    
        ml = SpectralEmbedding(**mtds["se"], affinity = "precomputed")
        visualize(distances, ml, legend=legend, method_name="Spectral Embedding", log=log)

def visualize(distances, ml = MDS(), legend:str ="Node", method_name = "MDS", log = False):

    positions = ml.fit_transform(distances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1])

    # Optional: Annotate the points
    for i, (x, y) in enumerate(positions):
        ax.text(x, y, f'{legend} {i}', fontsize=12)

    title = f'{legend} Visualization using {method_name}'

    ax.set_title(f'{legend} Visualization using {method_name}')
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.grid(True)

    if log:
        wandb.log({title: wandb.Image(fig)})

    plt.show()

# our trainer loss plot - flattened 

def visualize_loss(loss_record, num_desired_trajectories, trajectory_length, per_epoch = False):

    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(20, 6))

    if per_epoch:
        plt.plot(np.mean(loss_record, axis=1)/(num_desired_trajectories * (trajectory_length-1) ))
        plt.yscale('log')
        plt.title('Mean Training loss per epoch') # mean over the # trajectory and # steps in trajectory 
        plt.xlabel('Epochs')
    else:
        plt.plot(loss_record.reshape(-1))
        plt.yscale('log')
        plt.title('Training loss trajectory') # mean over the # trajectory and # steps in trajectory 
        plt.xlabel('Iterations')

        # Get the current y-ticks
        yticks = plt.yticks()[0]
        # Add vertical lines at each epochs
        for i in range(0, len(loss_record.reshape(-1)), num_desired_trajectories * (trajectory_length-1)):
            plt.axvline(x=i, color='red', linestyle='--', linewidth=0.5)
    
    plt.show()

def visualize_env(env):   
    adj = env.adj_matrix

    # Assuming you have the adjacency matrix stored in the variable "adj"
    G = nx.from_numpy_array(adj.numpy().astype(float), create_using = nx.DiGraph)

    # Draw the graph using seaborn
    # sns.set()
    seed = 70 # Consistent graph shape across runs
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos = pos, with_labels=True)
    # Show the plot
    plt.show()

    print(f"number of actions: {env.n_actions}")

    print("S-O mapping:")
    print(env.items)
    print("action-to-node:")
    print(env.action_to_node)
    print("node-to-action-matrix:")
    print(env.node_to_action_matrix)
    print("affordance / node - to action:")
    print(env.affordance)