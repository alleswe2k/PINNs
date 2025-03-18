import meshio
import matplotlib.pyplot as plt

# Read the mesh data from the .inp file
mesh = meshio.read("hole_middle_r01.inp")

# Extract node coordinates and elements (assuming 2D line elements)
nodes = mesh.points  # Node coordinates (N x 2 for 2D)
elements = mesh.cells[0].data  # Connectivity (elements, assuming line elements)

# Plot the mesh
fig, ax = plt.subplots()

# Plot the nodes
ax.scatter(nodes[:, 0], nodes[:, 1], c='r', label='Nodes')

# Plot the elements (assuming 2-node line elements)
for element in elements:
    x = nodes[element, 0]  # X-coordinates of the line's nodes
    y = nodes[element, 1]  # Y-coordinates of the line's nodes
    ax.plot([x[0], x[1]], [y[0], y[1]], 'k-', lw=0.5)  # Line between node 1 and node 2

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Mesh Visualization')

# Show grid
ax.grid(True)

# Show the plot
plt.show()
