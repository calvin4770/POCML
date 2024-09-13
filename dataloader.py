# Construct synthetic data and prepare data loader
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader 

# function to generate a random graph
def construct_random_subgraph(num_nodes, min_edges=2, max_edges=5):
    min_edges-=1
    max_edges-=1
    connections = np.random.uniform(size=(num_nodes, num_nodes))
    connections += np.roll(np.eye(num_nodes), 1, 1)  # make sure every node is connected
    connections -= np.eye(num_nodes)  # make sure no self-connections
    # set strongest connections
    sorted_connections = np.sort(connections)[:, ::-1]
    min_edges = min_edges
    max_edges = max_edges + 1
    min_edges = int(min_edges / 2 + 0.5)  # used to be min_edges // 2
    max_edges = int(max_edges / 2 + 0.5)
    indices = np.stack([np.arange(num_nodes), np.random.randint(min_edges, max_edges, num_nodes)])
    thresholds = sorted_connections[indices[0], indices[1]]
    connections = np.where(connections > thresholds[:, None], 1., 0.).astype(np.float32)
    connections = np.clip(connections + connections.T, 0, 1)
    return connections

# function to generate a random small world graph
def construct_small_world_graph(n_nodes_per_world=6, n_worlds=4):
    n_nodes = n_nodes_per_world * n_worlds
    connections = np.zeros((n_nodes, n_nodes))
    for world_i in range(n_worlds):
        connection = construct_random_subgraph(n_nodes_per_world)
        # add to global graph
        start_i = world_i * n_nodes_per_world
        end_i = (world_i+1) * n_nodes_per_world
        connections[start_i:end_i, start_i:end_i] = connection
        # connect to graph
        if world_i != 0:
            node_from = np.random.randint(start_i, end_i)
            node_to = node_from - n_nodes_per_world
            connections[node_from, node_to] = 1
            connections[node_to, node_from] = 1
    return connections

# function to generate a dead ends graph
def construct_dead_ends_graph():
    # construct the shell-like connectivity
    n_shells = 4
    n_neurons_per_shell = 6
    layers_with_circular_connection = [1]
    size = n_shells * n_neurons_per_shell
    connections = np.zeros((size, size)).astype(np.float32)
    eye = np.eye(n_neurons_per_shell)
    for i in range(n_shells):
        if i in layers_with_circular_connection:
            # add circular connection
            idx = i * n_neurons_per_shell
            connections[idx:idx+n_neurons_per_shell, idx:idx+n_neurons_per_shell] = np.roll(eye, 1, axis=1)

        # add connections to next shell
        if i + 1 < n_shells:
            # connect to outer shell
            idx = i * n_neurons_per_shell
            j = idx + n_neurons_per_shell
            connections[idx:idx+n_neurons_per_shell, j:j+n_neurons_per_shell] = eye

    connections = connections + connections.T
    return connections

def construct_two_tunnel_graph(tunnel_length=1, middle_tunnel_length=1):
    L = tunnel_length
    M = middle_tunnel_length
    num_nodes = tunnel_length * 2 + middle_tunnel_length + 4
    connections = np.zeros((num_nodes, num_nodes)).astype(np.float32)

    # first L -> upper tunnel, next L -> lower tunnel, next M -> middle tunnel
    # then upper corner, lower corner, upper end, lower end

    up_tun_head = 0
    up_tun_end = L - 1
    low_tun_head = L
    low_tun_end = L * 2 - 1
    mid_tun_head = L * 2
    mid_tun_end = L * 2 + M - 1
    up_corner = L * 2 + M
    low_corner = L * 2 + M + 1
    up_end = L * 2 + M + 2
    low_end = L * 2 + M + 3

    # construct upper and lower tunnels
    for i in range(tunnel_length-1):
        connections[up_tun_head + i, up_tun_head + i + 1] = 1
        connections[low_tun_head + i, low_tun_head + i + 1] = 1

    # construct middle tunnel
    for i in range(middle_tunnel_length-1):
        connections[mid_tun_head + i, mid_tun_head + i + 1] = 1

    # connect upper corner
    connections[up_corner, up_tun_head] = 1
    connections[up_corner, mid_tun_head] = 1

    # connect lower corner
    connections[low_corner, low_tun_head] = 1
    connections[low_corner, mid_tun_end] = 1

    # connect ends
    connections[up_end, up_tun_end] = 1
    connections[low_end, low_tun_end] = 1

    connections = connections + connections.T
    return connections

def construct_grid_graph(rows=5, cols=5):
    n_nodes = rows * cols
    connections = np.zeros((n_nodes, n_nodes)).astype(np.float32)
    for i in range(rows):
        for j in range(cols):
            idx = i * rows + j
            if i > 0:
                up_idx = (i-1) * rows + j
                connections[idx, up_idx] = 1
            if i < rows - 1:
                down_idx = (i+1) * rows + j
                connections[idx, down_idx] = 1
            if j > 0:
                left_idx = i * rows + j - 1
                connections[idx, left_idx] = 1
            if j < cols - 1:
                right_idx = i * rows + j + 1
                connections[idx, right_idx] = 1
    return connections

def construct_tree(levels=5):
    n_nodes = 2**levels - 1
    connections = np.zeros((n_nodes, n_nodes)).astype(np.float32)
    for i in range(n_nodes // 2 + 1):
        if 2*i+1 < n_nodes:
            connections[i, 2*i+1] = 1
            connections[i, 2*i+2] = 1
    connections += connections.T
    return connections

# constructs a k-regular graph (w.r.t. outgoing edges)
def construct_regular_graph(n_nodes, k, self_connections=False, replace=False):
    connections = np.zeros((n_nodes, n_nodes)).astype(np.float32)
    for i in range(n_nodes):
        if self_connections:
            idxs = np.arange(n_nodes)
        else:
            idxs = np.arange(n_nodes - 1)
            idxs[i:] += 1
        rand_idxs = np.random.choice(idxs, k, replace=replace)
        connections[i, rand_idxs] = 1
    return connections 

# Create a dataset of trajectories
class RandomWalkDataset(Dataset):
    def __init__(self,
                 adj_matrix,
                 trajectory_length,
                 num_trajectories,
                 items,
                 action_type="unique",
                 args=None,
                 permissible_starts=None):
        self.adj_matrix = adj_matrix
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.permissible_starts = permissible_starts
        self.edges, self.action_indices = edges_from_adjacency(adj_matrix, action_type=action_type, args=args)
        if permissible_starts is None:
            start_nodes = torch.randint(0, adj_matrix.size(0), (num_trajectories,)).tolist() # random start nodes
        else:
            start_nodes = np.random.choice(permissible_starts, self.num_trajectories).tolist()
        self.start_nodes = start_nodes
        self.data = []
        for node in start_nodes:
            trajectory = strict_random_walk(self.adj_matrix, node, self.trajectory_length, self.action_indices, items)
            self.data.append(torch.tensor([(x[0], x[1], x[2]) for x in trajectory]))
    def __len__(self):
        return self.num_trajectories  # Number of trajectories

    def __getitem__(self, idx):
        return self.data[idx]
    
    
# function to generate random walk trajectories on a given graph
def strict_random_walk(adj_matrix, start_node, length, action_indices, items):
    current_node = start_node
    trajectory = []
    for _ in range(length - 1):  # subtract 1 to account for the start node
        neighbors = torch.where(adj_matrix[current_node] > 0)[0].tolist()
        if not neighbors:
            break
        next_node = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
        trajectory.append((items[current_node], action_indices[(current_node, next_node)], items[next_node]))
        current_node = next_node
    return trajectory

# indexing each action for a given adjacency matrix
def edges_from_adjacency(adj_matrix, action_type='unique', args=None):
    # The input is a given random matrix's adjacency matrix
    # The outputs are:
        # edges: a list of pairs of (start node, end node) for each action
        # action_indices: a dictionary, each key is a pair of(start node, end node),
            # and its corresponding value is this action's index
    # For a pure on-line algorithm, this can also be done by assigning index to unseen actions
    # during random-walk on-line

    n = adj_matrix.shape[0]
    edges = []
    action_idx = 0
    action_indices = {}

    if action_type == 'unique':
        for i in range(n):
            for j in range(i+1, n):  # Only upper triangle
                if adj_matrix[i][j] != 0:
                    edges.append((i, j))
                    action_indices[(i, j)] = action_idx
                    action_idx += 1
                    edges.append((j, i))
                    action_indices[(j, i)] = action_idx
                    action_idx += 1
    elif action_type == 'regular':
        rng = np.random.default_rng()
        k = int(adj_matrix[0, :].sum())
        for i in range(n):
            idx = 0
            coloring = rng.permutation(k)
            for j in range(n):
                if adj_matrix[i, j] != 0:
                    edges.append((i, j))
                    action_indices[(i, j)] = coloring[idx]
                    idx += 1
    elif action_type == 'grid':
        # compute number of rows and columns
        rows = args["rows"]
        cols = args["cols"]
        for i in range(rows):
            for j in range(cols):
                idx = i * rows + j
                if i > 0:
                    up_idx = (i-1) * rows + j
                    edges.append((idx, up_idx))
                    action_indices[(idx, up_idx)] = 0
                if i < rows - 1:
                    down_idx = (i+1) * rows + j
                    edges.append((idx, down_idx))
                    action_indices[(idx, down_idx)] = 1
                if j > 0:
                    left_idx = i * rows + j - 1
                    edges.append((idx, left_idx))
                    action_indices[(idx, left_idx)] = 2
                if j < cols - 1:
                    right_idx = i * rows + j + 1
                    edges.append((idx, right_idx))
                    action_indices[(idx, right_idx)] = 3

    elif action_type == 'two tunnel':
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] == 1:
                    edges.append((i, j))
                    if j - i == 1: action_indices[(i, j)] = 2
                    if j - i == -1: action_indices[(i, j)] = 3
                    if j - i == 3: action_indices[(i, j)] = 1
                    if j - i == -3: action_indices[(i, j)] = 0
        # L = args["tunnel_length"]
        # M = args["middle_tunnel_length"]

        # # first L -> upper tunnel, next L -> lower tunnel, next M -> middle tunnel
        # # then upper corner, lower corner, upper end, lower end

        # up_tun_head = 0
        # up_tun_end = L - 1
        # low_tun_head = L
        # low_tun_end = L * 2 - 1
        # mid_tun_head = L * 2
        # mid_tun_end = L * 2 + M - 1
        # up_corner = L * 2 + M
        # low_corner = L * 2 + M + 1
        # up_end = L * 2 + M + 2
        # low_end = L * 2 + M + 3

        # # construct upper and lower tunnels
        # for i in range(L-1):
        #     edges.append((up_tun_head + i, up_tun_head + i + 1))
        #     action_indices[(up_tun_head + i, up_tun_head + i + 1)] = 0 # right
        #     edges.append((up_tun_head + i + 1, up_tun_head + i))
        #     action_indices[(up_tun_head + i + 1, up_tun_head + i)] = 1 # left
        #     edges.append((low_tun_head + i, low_tun_head + i + 1))
        #     action_indices[(low_tun_head + i, low_tun_head + i + 1)] = 0 # right
        #     edges.append((low_tun_head + i + 1, low_tun_head + i))
        #     action_indices[(low_tun_head + i + 1, low_tun_head + i)] = 1 # left

        # # construct middle tunnel
        # for i in range(M-1):
        #     edges.append((mid_tun_head + i, mid_tun_head + i + 1))
        #     action_indices[(mid_tun_head + i, mid_tun_head + i + 1)] = 2 # down
        #     edges.append((mid_tun_head + i + 1, mid_tun_head + i))
        #     action_indices[(mid_tun_head + i + 1, mid_tun_head + i)] = 3 # up

        # # connect upper corner
        # edges.append((up_corner, up_tun_head))
        # action_indices[(up_corner, up_tun_head)] = 0 # right
        # edges.append((up_tun_head, up_corner))
        # action_indices[(up_tun_head, up_corner)] = 1 # left
        # edges.append((up_corner, mid_tun_head))
        # action_indices[(up_corner, mid_tun_head)] = 2 # down
        # edges.append((mid_tun_head, up_corner))
        # action_indices[(mid_tun_head, up_corner)] = 3 # up

        # # connect lower corner
        # edges.append((low_corner, low_tun_head))
        # action_indices[(low_corner, low_tun_head)] = 0 # right
        # edges.append((low_tun_head, low_corner))
        # action_indices[(low_tun_head, low_corner)] = 1 # left
        # edges.append((low_corner, mid_tun_end))
        # action_indices[(low_corner, mid_tun_end)] = 3 # up
        # edges.append((mid_tun_end, low_corner))
        # action_indices[(mid_tun_end, low_corner)] = 2 # down

        # # connect ends
        # edges.append((up_end, up_tun_end))
        # action_indices[(up_end, up_tun_end)] = 0 # right
        # edges.append((up_tun_end, up_end))
        # action_indices[(up_tun_end, up_end)] = 1 # left
        # edges.append((low_end, low_tun_end))
        # action_indices[(low_end, low_tun_end)] = 0 # right
        # edges.append((low_tun_end, low_end))
        # action_indices[(low_tun_end, low_end)] = 1 # left
    elif action_type == "tree":
        for i in range(n):
            if 2*i+1 < n:
                j1 = 2*i+1
                j2 = 2*i+2
                edges.append((i, j1))
                action_indices[(i, j1)] = 1 # left
                edges.append((i, j2))
                action_indices[(i, j2)] = 2 # right
                edges.append((j1, i))
                action_indices[(j1, i)] = 0 # up
                edges.append((j2, i))
                action_indices[(j2, i)] = 0 # up

    return edges, action_indices

class GraphEnv():
    def __init__(
            self,
            n_items=10, # number of possible observations
            env='random', 
            batch_size=15, 
            num_desired_trajectories=10, 
            device=None, 
            unique=False, # each state is assigned a unique observation if true
            args = None
        ):
        if env == 'random':
            n_nodes = args["n_nodes"]
            self.adj_matrix = construct_random_subgraph(n_nodes, 2, 5)
        elif env == 'small world':
            self.adj_matrix = construct_small_world_graph()
        elif env == 'dead ends':
            self.adj_matrix = construct_dead_ends_graph()
        elif env == 'grid':
            rows = args["rows"]
            cols = args["cols"]
            self.adj_matrix = construct_grid_graph(rows, cols)
            self.n_actions = 4
        elif env == 'two tunnel':
            # tunnel_length = args["tunnel_length"]
            # middle_tunnel_length = args["middle_tunnel_length"]
            # self.adj_matrix = construct_two_tunnel_graph(
            #     tunnel_length=tunnel_length, middle_tunnel_length=middle_tunnel_length)
            self.adj_matrix = construct_grid_graph(3, 3)
            # nodes 4 and 7 are blocked
            self.adj_matrix[4, :] = 0
            self.adj_matrix[:, 4] = 0
            self.adj_matrix[7, :] = 0
            self.adj_matrix[:, 7] = 0
            self.n_actions = 4
        elif env == 'regular':
            n_nodes = args["n_nodes"]
            self.n_actions = k = args["k"]
            self.adj_matrix = construct_regular_graph(n_nodes, k)
        elif env == "tree":
            levels = args["levels"]
            self.adj_matrix = construct_tree(levels)
        
        self.env = env
        self.args = args

        self.action_type = "unique"
        if self.env in ["regular", "two tunnel", "grid", "tree"]:
            self.action_type = self.env

        self.adj_matrix = torch.tensor(self.adj_matrix)
        self.size = self.adj_matrix.shape[0] # number of nodes
        self.affordance, self.node_to_action_matrix,\
        self.action_to_node = node_outgoing_actions(self.adj_matrix, action_type=self.action_type, args=self.args)
        
        self.affordance = {k: torch.tensor(v).to(device)\
                           for k, v in self.affordance.items()}
        self.node_to_action_matrix = self.node_to_action_matrix.to(device)
        self.action_to_node = {k: torch.tensor(v).to(device) \
                            for k, v in self.action_to_node.items()}
        
        self.unique = unique
        if not unique:
            self.n_items = n_items
        else:
            self.n_items = self.size
        self.batch_size = batch_size
        self.num_desired_trajectories = num_desired_trajectories
        self.start_state_idx = np.random.randint(0, self.n_items) # fixed start state for trajectories if enabled
        self.populate_graph()

    # uniformly random observations or identity (unique)
    # obs_state_map = array of size [n_states] storing obs indices
    def populate_graph(self, obs_state_map=None):
        if obs_state_map is None:
            if self.unique:
                self.items = torch.arange(0, self.n_items)
            else:
                self.items = (torch.rand(self.size) * self.n_items).to(torch.int32)
        else:
            self.items = obs_state_map

    def gen_dataset(self, batch_size=None, num_desired_trajectories=None):
        if batch_size is None:
            batch_size = self.batch_size
        if num_desired_trajectories is None:
            num_desired_trajectories = self.num_desired_trajectories

        action_type = "unique"
        if self.env in ["regular", "two tunnel", "grid", "tree"]:
            action_type = self.env
            
        # TODO: fixed start
        self.dataset = RandomWalkDataset(
            self.adj_matrix,
            batch_size,
            num_desired_trajectories,
            self.items,
            action_type=action_type,
            args=self.args,
            permissible_starts=([0, 1, 2, 3, 5, 6, 8] if self.env == 'two tunnel' else None)
        )
        if self.env not in ["regular", "two tunnel", "grid"]:
            self.n_actions = len(self.dataset.action_indices)
        return self.dataset.data

    def populate_graph_preset(self):
        if self.env == 'two tunnel':
            # L = self.tunnel_length
            # M = self.middle_tunnel_length
            # self.items = torch.zeros(self.size)
            # self.items[L*2:L*2+M] = 1
            # self.items[L*2+M] = 2
            # self.items[L*2+M+1] = 3
            # self.items[L*2+M+2] = 4
            # self.items[L*2+M+3] = 5
            self.items = torch.tensor([0, 1, 2, 3, 4, 3, 5, 4, 6])
        
def node_outgoing_actions(adj_matrix, action_type="unique", args=None):
    # This function creates several look-up tables for later computation's convecience
    edges, action_indices = edges_from_adjacency(adj_matrix, action_type=action_type, args=args)
    # Use an action index as a key, retrieve its (start node, end node)
    inverse_action_indices = {v: k for k, v in action_indices.items()}
    # Given a node as a key, retrieve all of its available outgoing actions' indexes.
    node_actions = {}
    # Given a pair of (start node, end node), get the action index.
    # Since a index can be 0, this matrix is initialized to be a all -1.
    node_to_action_matrix = -1*torch.ones_like(adj_matrix)
    for edge in edges:
        node_from, node_to = edge
        if node_from not in node_actions:
            node_actions[node_from] = []
        node_actions[node_from].append(action_indices[edge])
        node_to_action_matrix[node_from][node_to] = action_indices[edge]  
    return node_actions, node_to_action_matrix.long(), inverse_action_indices