NETENSORFLOW_CLASSES = dict()


def register_netensorflow_class(class_to_append):
    NETENSORFLOW_CLASSES.update({class_to_append.__name__: class_to_append})
    return class_to_append
