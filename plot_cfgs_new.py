import angr
import networkx as nx
import matplotlib.pyplot as plt

# Load the binary
project = angr.Project('../bin/ls', auto_load_libs=False)

# Get the CFG
cfg = project.analyses.CFG()

# Create a directed graph from the CFG
graph = cfg.graph 

# Visualize the CFG
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(graph, seed=42)  # Set a seed for reproducibility
nx.draw(graph, pos, with_labels=False, node_size=10, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
plt.title('Control Flow Graph (CFG)', fontsize=15)
plt.show()
plt.savefig("demo_ls_cfg.png")