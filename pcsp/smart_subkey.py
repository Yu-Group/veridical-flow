class SmartSubkey:
    def __init__(self, subkey, origin: str):
        '''
        Params
        -------
        key:
            subkey corresponding to a ModuleSet module
        origin: str
            name of the origin ModuleSet of this subkey
        '''
        self.subkey = subkey
        self.origin = origin
    
    def __repr__(self) -> str:
        return f"{repr(self.subkey)}-{repr(self.origin)}"
    
    def __eq__(self, o: object):
        if isinstance(o, self.__class__):
            return self.subkey == o.subkey and self.origin == o.origin
        return False
    
    def __hash__(self):
        return hash(self.subkey) ^ hash(self.origin)