"""Defines a parameter from some origin Vset
"""
class Subkey:

    def __init__(self, value, origin: str, output_matching: bool = False):
        """
        Parameters
        ----------
        value: Any
            subkey value corresponding to a Vset module
        origin: str
            name of the origin Vset of this Subkey
        output_matching: bool (optional), default False
            inherited from the Vset where the Subkey is created
        """
        self.value = value
        self.origin = origin
        self.output_matching = output_matching
        # sep_dicts_id identifies the particular call to sep_dicts() that this
        # key's dictionary went through (if any).
        self.sep_dicts_id = None

    def is_matching(self):
        """Checks if subkey should be matched in other Vsets
        """
        return self.output_matching or self.sep_dicts_id is not None

    def matches_sep_dict_id(self, other: object):
        """Helper to match Subkey by _sep_dict_id
        """
        if isinstance(other, self.__class__):
            return self.sep_dicts_id is not None \
                   and self.sep_dicts_id == other.sep_dicts_id
        return False

    def matches(self, other: object):
        """When Subkey matching is required, determines if this Subkey is compatible
        with another, meaning that the origins and values match, and either the
        _sep_dicts_id matches or both Subkeys have _output_matching True.
        """
        if isinstance(other, self.__class__):
            # they're both matching
            cond0 = self.is_matching() and other.is_matching()
            # value and origins match
            cond1 = self.value == other.value and self.origin == other.origin
            # sep_dicts_id matches
            cond2 = self.sep_dicts_id == other.sep_dicts_id or \
                    (self.output_matching and other.output_matching)
            return cond0 and cond1 and cond2
        return False

    def mismatches(self, other: object):
        """When Subkey matching is required, determines if this Subkey and another are
        a bad match, meaning either:

        1. output_matching is True, origin is same, value is different
        2. output_matching is False, sep_dicts_id is same and not None, origin
           is same, value is different
        """
        if isinstance(other, self.__class__):
            # one of the two keys is output_matching
            cond0 = self.output_matching or other.output_matching
            # neither key is output_matching but sep_dict_ids not None and match
            cond1 = not cond0 and self.matches_sep_dict_id(other)
            # origins match and values mismatch
            cond2 = self.origin == other.origin and self.value != other.value
            return (cond0 or cond1) and cond2
        return True

    def __eq__(self, other: object):
        """Mainly used for testing purposes.
        """
        if isinstance(other, self.__class__):
            # value and origins match
            return self.value == other.value and self.origin == other.origin
        return False

    def __repr__(self):
        return str(self.value)

    def __hash__(self):
        """Mainly used for testing purposes.
        """
        return hash(self.value) ^ hash(self.origin) ^ hash(self.output_matching)
