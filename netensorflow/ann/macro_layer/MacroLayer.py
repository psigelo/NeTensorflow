class MacroLayer(object):
    def __init__(self, layers_structure=None):
        self.layers_structure_list = None
        if isinstance(layers_structure, list):
            self.layers_structure_list = layers_structure
