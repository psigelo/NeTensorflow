NETENSORFLOW_CLASSES = list()


def register_netensorflow_class(class_to_append):
    NETENSORFLOW_CLASSES.append(class_to_append.__name__)
