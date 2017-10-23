NEURON_LAYER_REGISTERED = list()

def register_neuron_layer(class_to_append):
    NEURON_LAYER_REGISTERED.append(class_to_append.__name__)