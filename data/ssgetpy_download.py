import ssgetpy

if __name__ == '__main__':
    """
    Script to download datasets from https://sparse.tamu.edu/ (Suite Sparse Matrix Collection) by dataset name.
    """
    name = "Pres_Poisson"
    res = ssgetpy.search(name)[0]
    res.download(destpath="../data/compressed/")
