# Use the official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install Python and pip
RUN apt-get update && \
    apt-get install -y python3-pip file

# Install torch
RUN pip3 install torch

# Install torch-geometric
RUN pip3 install torch-geometric

# Install angr
RUN pip3 install angr

RUN pip3 install data

# Install angr-utils and required packages
RUN pip3 install angr-utils

RUN apt install -y graphviz


# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set the default command to run when the container starts
CMD ["bash"]
