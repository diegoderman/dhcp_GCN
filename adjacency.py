import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, save_npz
import matplotlib.pyplot as plt


def gifti_surface_to_adjacency_matrix(gifti_file, output_adj_path=None):
    """
    Load a GIFTI surface and generate the adjacency matrix.
    
    Parameters:
    -----------
    gifti_file : str
        Path to the .surf.gii GIFTI surface file.
    output_adj_path : str, optional
        If provided, saves the sparse adjacency matrix to this path in npz format.
    
    Returns:
    --------
    adj_matrix : csr_matrix
        Sparse vertex adjacency matrix.
    """
    # Load the GIFTI file
    gii = nib.load(gifti_file)
    
    # Extract coordinates and triangle indices
    coords = gii.darrays[0].data
    faces = gii.darrays[1].data

    # Number of vertices
    n_vertices = coords.shape[0]

    # Collect edges
    rows = []
    cols = []

    for face in faces:
        # Each triangle connects 3 vertices: (i,j), (j,k), (k,i)
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            v1 = face[i]
            v2 = face[j]
            rows.extend([v1, v2])
            cols.extend([v2, v1])  # undirected edge

    data = np.ones(len(rows), dtype=np.uint8)
    
    # Build sparse adjacency matrix
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()

    # Optionally save
    if output_adj_path:
        from scipy.sparse import save_npz
        save_npz(output_adj_path, adj_matrix)
        print(f"Adjacency matrix saved to {output_adj_path}")
    
    return adj_matrix

def plot_adjacency_matrix(adj_matrix, title="Adjacency Matrix", figsize=(6, 6), markersize=0.5):
    """
    Plot the sparsity pattern of a sparse adjacency matrix.
    
    Parameters:
    -----------
    adj_matrix : scipy.sparse matrix
        Sparse adjacency matrix (e.g., from gifti surface).
    title : str
        Title for the plot.
    figsize : tuple
        Size of the figure.
    markersize : float
        Dot size in the plot.
    """
    plt.figure(figsize=figsize)
    plt.spy(adj_matrix, markersize=markersize)
    plt.title(title)
    plt.xlabel("Vertex index")
    plt.ylabel("Vertex index")
    plt.tight_layout()
    plt.show()


# Start main
if __name__ == "__main__":
    # Example usage
    hemi = "left"
    gifti_file = '/data/dderman/baby_ICA/surf_symatlas/week-40_hemi-' + hemi + '_space-dhcpSym_dens-32k_midthickness.surf.gii'
    output_adj_path = './adjacency_matrix-' + hemi + '.npz'

    adj_lh = gifti_surface_to_adjacency_matrix(gifti_file, output_adj_path)

    hemi = "right"
    gifti_file = '/data/dderman/baby_ICA/surf_symatlas/week-40_hemi-' + hemi + '_space-dhcpSym_dens-32k_midthickness.surf.gii'
    output_adj_path = './adjacency_matrix-' + hemi + '.npz'

    adj_rh = gifti_surface_to_adjacency_matrix(gifti_file, output_adj_path)

    # Combine into one adjacency matrix (block-diagonal)
    adj_both = block_diag((adj_lh, adj_rh), format="csr")
    # save combined adjacency matrix
    combined_adj_path = './adjacency_matrix_combined.npz'
    save_npz(combined_adj_path, adj_both)
    # plot
    plot_adjacency_matrix(adj_both)

    print("Combined adjacency shape:", adj_both.shape)