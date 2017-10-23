class ANN(object):
    def __init__(self, macro_layers=None, tf_session=None):
        self.macro_layers = macro_layers
        self.tf_session = tf_session
        self.last_layer = None
        self.first_layer = None

    def connect_and_initialize(self):
        layers_refs = list()

        # first get all layers
        for layer_structures in self.macro_layers.layers_structure_list:
            for layers in layer_structures.layers:
                layers_refs.append(layers)

        # Connect all layers
        for it in range(1, len(layers_refs)):  # Starting from second layer
            layers_refs[it].connect(layers_refs[it-1].get_input_amount(), layers_refs[it-1].get_tensor())

        self.last_layer = layers_refs[-1]
        self.first_layer = layers_refs[1]

    def run(self):
        input_tensor = self.first_layer.get_input_tensor()
        output_tensor = self.last_layer.get_tensor()
        import random
        self.tf_session.run(output_tensor, {input_tensor, [random.randint(0,5) for i in range(100)]})