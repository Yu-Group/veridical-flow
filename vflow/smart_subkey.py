class SmartSubkey:
    def __init__(self, subkey, origin: str, output_matching: bool=False):
        '''
        Params
        -------
        key:
            subkey corresponding to a Vset module
        origin: str
            name of the origin Vset of this subkey
        '''
        self.subkey = subkey
        self.origin = origin
        self._output_matching = output_matching
    
    def is_matching(self):
        return self._output_matching

    def __repr__(self) -> str:
        return self.subkey

    def __eq__(self, o: object):
        if isinstance(o, self.__class__):
            return self.subkey == o.subkey and self.origin == o.origin \
                    and self._output_matching == o._output_matching
        return False
    
    def __lt__(self, o: object):
        if isinstance(o, self.__class__):
            return self.subkey < o.subkey
        return False

    def __hash__(self):
        return hash(self.subkey) ^ hash(self.origin) ^ hash(self._output_matching)
