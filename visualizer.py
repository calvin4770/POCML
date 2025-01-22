from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb

import networkx as nx

def pca_visualize(model, env, log=False, show=True):
    pca = PCA(2)
    Q_pca = pca.fit_transform(model.Q.T)
    V_pca = pca.transform(model.V.T)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Q_pca[:, 0], Q_pca[:, 1])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    colors = ["red", "green", "blue", "yellow"]
    for i, (x, y) in enumerate(Q_pca):
        ax.text(x, y, f'State {i}', fontsize=16)
        actions = env.affordance[i].tolist()
        for a in actions:
            ax.plot([Q_pca[i, 0], Q_pca[i, 0] + V_pca[a, 0]], [Q_pca[i, 1], Q_pca[i, 1] + V_pca[a, 1]], c=colors[a])
    ax.grid(True)

    if log:
        wandb.log({"pca": wandb.Image(fig)})

    if show:
        plt.show()
    else:
        plt.close(fig)

# our trainer loss plot - flattened 

def visualize_loss(loss_record, num_desired_trajectories, trajectory_length, per_epoch = False, show = True):

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

    if show: 
        plt.show()
    else:
        plt.close()

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