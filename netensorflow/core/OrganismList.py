from netensorflow.core.Organism import Organism


class OrganismList(object):
    def __init__(self):
        def __init__(self, *args):
            self.organisms_list = list()

        def __iter__(self, ):
            return self.species.__iter__()

        def __len__(self):
            return len(self.species)

        def append(self, item):
            if not isinstance(item, Organism):
                raise (ValueError, "item must be a specie")
            self.species.append(item)