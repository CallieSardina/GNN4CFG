# Open the file for reading
with open('bin_binaries.txt', 'r') as file:
    # Read each line and extract text before ":"
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespaces
        if ':' in line:
            text = line.split(':')[0]
            text = './bin/' + text[2:]
            print(text)