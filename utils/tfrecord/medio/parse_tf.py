import pathlib as pl


def fetch_paths(dir_tf, pattern):
    """ 
    This function is highly data set dependent.
    output: dictionary containing subject as key and reorganized sequences and labels
     """

    # convert to paths
    if not isinstance(dir_tf, pl.Path):
        path_tf = pl.Path(dir_tf)
    else:
        path_tf = dir_tf

    path_list = []

    for subject, files in _parse_tf_gen(path_tf):
        for file in files:
            if pattern in file.name:
                path_list.append(str(file))

    return path_list


def _parse_tf_gen(path):
    for subject in path.iterdir():
        if subject.is_dir():
            # tfrecord
            files = subject.rglob('*.tfrecord')
            # rglob can read all file endet with .tfrecord
            yield subject, files
