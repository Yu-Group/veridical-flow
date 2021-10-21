class Subkey:
    def __init__(self, value, origin: str, output_matching: bool=False):
        '''
        Params
        -------
        value:
            subkey value corresponding to a Vset module
        origin: str
            name of the origin Vset of this Subkey
        output_matching: inherited from the Vset where the Subkey is created
        '''
        self.value = value
        self.origin = origin
        self._output_matching = output_matching
        # _sep_dicts_id identifies the particular call to sep_dicts() that this
        # key's dictionary went through (if any). Supersedes _output_matching.
        self._sep_dicts_id = None

    def is_matching(self):
        return self._output_matching or self._sep_dicts_id is not None

    def matches(self, o: object):
        '''
        Determines if this Subkey is compatible with another for the purpose of
        combining data dictionaries and matching data to modules.
        '''
        if isinstance(o, self.__class__):
            # they're both matching
            cond0 = self.is_matching() and o.is_matching()
            # value and origins match
            cond1 = self.value == o.value and self.origin == o.origin
            # _sep_dicts_id matches
            cond2 = self._sep_dicts_id == o._sep_dicts_id
            return cond0 and cond1 and cond2
        return False

    def mismatches(self, o: object):
        '''
        Determines if this Subkey and another are a bad match, meaning they have
        the same origin but different values when both are matching.
        '''
        if isinstance(o, self.__class__):
            # one of the two keys is matching
            cond0 = self.is_matching() or o.is_matching()
            # origins match
            cond1 = self.origin == o.origin
            # values or _sep_dicts_id mismatch
            cond2 = self.value != o.value # \
                # or self._sep_dicts_id != o._sep_dicts_id
            return cond0 and cond1 and cond2
        return True

    def __eq__(self, o: object):
        '''
        Mainly used for testing purposes.
        '''
        if isinstance(o, self.__class__):
            # value and origins match
            return self.value == o.value and self.origin == o.origin
        return False

    def __repr__(self) -> str:
        return self.value

    def __hash__(self):
        '''
        Mainly used for testing purposes.
        '''
        return hash(self.value) ^ hash(self.origin) ^ hash(self._output_matching)
