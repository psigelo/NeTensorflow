import sys

from netensorflow.core.SpeciesList import SpeciesList


class Life(object):
    def __init__(self, parameters):
        self.__parameters = None
        self.species_list = SpeciesList()
        self.parameters = parameters

    def add_specie(self, specie, force=False):
        if force or len(self.species_list) < self.parameters['max_amount_species']:
            self.species_list.append(specie)
        else:
            raise(BufferError, "can not have more species, try param force=True or a larger max_amount_species")

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, params):
        if not self.check_params(params):
            raise (ValueError, "params are not correctly setted")
        self.__parameters = params
        self.species_list.parameters = params

    def epoch(self):
        if len(self.species_list) == 0:
            raise (AttributeError, "Species not initialized")

    @staticmethod
    def check_params(params):
        try:
            if not isinstance(params['max_amount_species'], int):
                print('max_amount_species must be integer')
                return False
            if params['max_amount_species'] <= 0:
                print('max_amount_species must be greater than 0')
                return False

        except KeyError as e:
            print("Parameter not defined: {0}".format(e))
            return False
        except:
            print("Unexpected error:", sys.exc_info()[0])
            return False

        return True  # not error detected
