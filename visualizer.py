from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb

def visualize(distances, legend:str ="Node", title:str = "Graph Visualization using MDS", log = False):
    mds = MDS(n_components =2)
    positions = mds.fit_transform(distances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1])

    # Optional: Annotate the points
    for i, (x, y) in enumerate(positions):
        ax.text(x, y, f'{legend} {i}', fontsize=12)

    ax.set_title('Graph Visualization using MDS')
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

