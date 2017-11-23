class Specie(object):
    def __init__(self):
        self.__parameters = None
        self.organisms = None  # ToDo: Must be a OrganismList class
        self.model = None  # tensorflow-like model

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        if not self.check_params(params):
            raise (ValueError, "params are not correctly setted")
        self.organisms.parameters = params

    @staticmethod
    def check_params(params):
        return True  # ToDo

