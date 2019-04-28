def dynamic_importer(name):
    """
    Dynamically imports modules / classes
    """
    module = __import__(name)

    return module