from netensorflow.core.Specie import Specie


class SpeciesList(object):
    def __init__(self):
        self.species = list()
        self.__parameters = None

    def __iter__(self):
        return self.species.__iter__()

    def __len__(self):
        return len(self.species)

    def append(self, item):
        if not isinstance(item, Specie):
            raise(ValueError, "item must be a specie")
        self.species.append(item)

    @property
    def parameters(self):
        if self.__parameters is None:
            raise (AttributeError, "Parameters not setted previously")
        return self.__parameters

    @parameters.setter
    def parameters(self, params):  # simply deliver
        for specie in self.species:
            specie.parameters = params
