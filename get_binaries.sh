#!/bin/bash
cd /bin

find . -type f -exec file {} + | grep ELF
