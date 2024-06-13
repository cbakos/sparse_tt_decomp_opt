import tarfile


if __name__ == '__main__':
    """
    Script to extract compressed tar files.
    """
    name = "Pres_Poisson"
    # open file
    file = tarfile.open('compressed/{}.tar.gz'.format(name))
    # extracting file
    file.extractall('../data')
    file.close()
