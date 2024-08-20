import torch
import numpy as np

class Graph():

    def __init__(self, num_nodes=14, strategy='functional', tau=0.5):
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.tau = tau  # Threshold for adjacency matrix
        self.adjacency_matrix = self.build_adjacency_matrix()

    def build_adjacency_matrix(self):
        if self.strategy == 'uni-label':
            return self.uni_label_partition()
        elif self.strategy == 'dual-label':
            return self.dual_label_partition()
        elif self.strategy == 'functional':
            return self.functional_partition()
        else:
            raise ValueError("Unknown partition strategy.")

    def uni_label_partition(self):
        A = np.ones((self.num_nodes, self.num_nodes))
        # np.fill_diagonal(A, 1)
        return A


    def dual_label_partition(self):
    # Initialize adjacency matrix with zeros
        A = np.zeros((self.num_nodes, self.num_nodes))
        root_node = 8  # Assume the first node is the root for simplicity
        
        for i in range(self.num_nodes):
            if i == root_node:
                A[i, :] = 1  # Connect root node to all others
                A[:, i] = 1  # Ensure symmetry
            elif A[root_node, i] == 1:  # Check if node i is directly connected to root
                A[root_node, i] = 1
                A[i, root_node] = 1

        # Optionally include self-loops
        np.fill_diagonal(A, 1)

        return A



    def functional_partition(self):
        # Divide into functional groups: root, TBR, SBR
        A = np.zeros((self.num_nodes, self.num_nodes))
        
        # Assign nodes based on known EEG mapping to brain regions
        root_node = 9  # Choose P1 as a root node for this example (adjust as needed)

        # Assign indices for Thinking Brain Region (TBR)
        tbr_nodes = [0, 1, 4, 5, 6, 7]  

        # Assign indices for Sensory Brain Region (SBR)
        sbr_nodes = [2, 3, 8, 9, 10, 11, 12, 13]  

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    A[i, j] = 1  # Self-loops
                elif i == root_node or j == root_node:
                    A[i, j] = 1  # Connect root to all nodes
                elif (i in tbr_nodes and j in tbr_nodes) or (i in sbr_nodes and j in sbr_nodes):
                    A[i, j] = 1  # Within-region connections

        A[A < self.tau] = 0  # Apply threshold
        return A


    def pli(self, data):
        """
        Compute the Phase Lag Index (PLI) for EEG signals.
        
        Args:
            data (torch.Tensor): Input signal, shape (batch, frames, nodes).
        
        Returns:
            torch.Tensor: PLI matrix, shape (batch, nodes, nodes).
        """
        N, T, V = data.size()
        
        # Perform Hilbert transform to get instantaneous phase
        analytic_signal = torch.view_as_complex(torch.fft.rfft(data, dim=1))
        phase_data = torch.angle(analytic_signal)
        
        pli_matrix = torch.zeros((N, V, V))
        
        for i in range(V):
            for j in range(i + 1, V):
                # Compute phase difference
                phase_diff = phase_data[:, :, i] - phase_data[:, :, j]
                
                # Compute PLI based on the formula
                pli = torch.abs(torch.mean(torch.sign(torch.sin(phase_diff)), dim=1))
                
                pli_matrix[:, i, j] = pli
                pli_matrix[:, j, i] = pli
        
        return pli_matrix


    def get_size(self, node_num):
        # Assuming the adjacency matrix size is equal to the number of nodes
        return (self.adjacency_matrix.shape[0], self.adjacency_matrix.shape[1])

    def __call__(self, x):
        # Expand the adjacency matrix to match batch size and number of partitions
        # Assuming `x` is of shape (batch_size, channels, time_steps, num_nodes)
        batch_size = x.size(0)
        k = 1  # Number of partitions, change as needed for your partitioning strategy
        
        A_expanded = np.expand_dims(self.adjacency_matrix, axis=0)  # Add a new axis for batch
        A_expanded = np.expand_dims(A_expanded, axis=0)  # Add a new axis for partitions
        
        A_expanded = np.tile(A_expanded, (batch_size, k, 1, 1))  # Repeat for each batch and partition
        
        return torch.tensor(A_expanded, dtype=torch.float32)



def normalize_adjacency(A):
    Dl = torch.sum(A, dim=1, keepdim=True)
    Dl[Dl <= 0] = 1
    D_inv_sqrt = torch.pow(Dl, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    normalized_A = D_inv_sqrt * A * D_inv_sqrt.transpose(-1, -2)
    return normalized_A

if __name__ == '__main__':


    g = Graph(num_nodes=14, strategy='functional', tau=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(64, 1, 128*5, 14)  # Sample input data with (batch_size, channels, time_steps, nodes)
    adj_matrix = g(a)  # Obtain the adjacency matrix based on the strategy
    
    print("Adjacency Matrix:")
    print(adj_matrix.shape)
