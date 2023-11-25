# GNN4CFG
Steps:


1: Run script to get list of binary files from ./bin directory and route the output to bin_binaries.txt file.

./get_binaries.sh > bin_binaries.txt

2: Format binary filenames in bin_binaries.txt to be processed correctly.

python3 process_binaries_txt.py > bin_binaries_list.txt

3: Run gcn_and_load.py to generate the .pt files for each binary in above list.

4: Run test4.py to train & eval data using GCN in gcn_temp.py.

To use the dockerfile:
1. Build the image 
   docker build -t your_chosen_image_name .
2. Run the container
  docker run -it your_chosen_image_name
3. Once done running stop and then remove the container

In the dockerfile, I install all the necessary packages and then copy the files from the current directory on the host machine to the /app directory in the container

## Required:

GraphViz needs to be installed on the system that is running anything with angr_utils.

