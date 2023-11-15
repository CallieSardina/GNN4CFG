# GNN4CFG
Steps:


1: Run script to get list of binary files from ./bin direcory and route the output to bin_binaries.txt file.

./get_binaries.sh > bin_binaries.txt

2: Format binary fileneames in bin_binaries.txt to be processed correctly.

python3 process_binaries_txt.py > bin_binaries_list.txt

3: Run gcn_and_load.py to generate the .pt files for each binary in above list.

4: Run test4.py to train & eval data using GCN in gcn_temp.py.

