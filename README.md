# GNN4CFG
Steps:


1: Run script to get list of binary files from ./bin direcory and route the output to bin_binaries.txt file.

./get_binaries.sh > bin_binaries.txt

2: Format binary fileneames in bin_binaries.txt to be processed correctly.

python3 process_binaries_txt.py > bin_binaries_list.txt

3: Run python code to generate the CFGs from the files in bin_binaries.txt and output graphs to cfg_output directory (We can then load these graphs for use in project).

python3 generate_cfg.py

