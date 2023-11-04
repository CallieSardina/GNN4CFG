import angr
import os
import json
import networkx as nx
import pickle 

def generate_cfg(binary_path):
    # Load the binary
    project = angr.Project(binary_path, auto_load_libs=False, main_opts={'backend': 'blob', 'arch': 'amd64'})

    # Get the CFG
    cfg = project.analyses.CFG()

    print("This is the graph:", cfg.graph)
    print("It has %d nodes and %d edges" % (len(cfg.graph.nodes()), len(cfg.graph.edges())))
    # Convert CFG to a JSON serializable format
    # cfg_data = nx.readwrite.json_graph.node_link_data(cfg.graph)

     
    pickle.dump(cfg.graph, open('cfg_output/' + binary_path[5:] + '.txt', 'wb'))

# Read the list of binary files and generate CFGs
#with open("binary_files.txt", "r") as file:
with open("bin_binaries_list.txt", "r") as file:
    binary_paths = file.read().splitlines()

for binary_path in binary_paths:
    generate_cfg(binary_path)
