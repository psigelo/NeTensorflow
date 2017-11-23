class Organism(object):
    def __init__(self):
        self.neural_network = None
        self.__parameters = None

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        if not self.check_params(params):
            raise (ValueError, "params are not correctly setted")
        self.neural_network.parameters = params

    @staticmethod
    def check_params(params):
        return True  # ToDo



