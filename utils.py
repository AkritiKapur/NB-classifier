def read_file_in_list(file_name):
    """
        Reads a file and returns the line in list format.
    :param file_name:
    :return: {List} list of lines(list of words)
    """
    lines = []
    with open(file_name, 'r') as f:
        read_lines = f.readlines()
        for line in read_lines:
            lines.append(line.split())

    return lines
