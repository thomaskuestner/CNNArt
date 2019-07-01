import pathlib as pl


def fetch_paths(dir_tf, patterns, testpatients, ismask=False):
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

    iSubject = 0
    for subject, files in _parse_tf_gen(path_tf):
        if iSubject in testpatients:
            iSubject += 1
            continue
        for file in files:
            for pattern in patterns:
                if ismask:
                    searchpattern = pattern.pathdata + '_mask'
                else:
                    searchpattern = pattern.pathdata

                if searchpattern == file.name:  # exact matching to avoid confusion to segmentation masks
                    path_list.append(str(file))
        iSubject += 1

    return path_list


def _parse_tf_gen(path):
    for subject in path.iterdir():
        if subject.is_dir():
            # tfrecord
            files = subject.rglob('*.tfrecord')
            # rglob can read all file endet with .tfrecord
            yield subject, files
