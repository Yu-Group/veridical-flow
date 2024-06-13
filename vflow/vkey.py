"""Vdict key tuple wrapper class
"""
from typing import Tuple, Union

from vflow.utils import combine_keys

class Vkey:

    def __init__(self, subkeys, origin: str, method: str):
        """
        Parameters
        ----------
        subkeys: tuple
            Tuple of Subkey objects to associate with this Vkey.
        origin: str
            String attribute that identifies the Vset that created this Vkey.
        method: str
            String attribute that identifies the Vset method that was called to create
            this Vkey.
        """
        self._subkeys = subkeys
        self.origin = origin
        self.method = method

    def subkeys(self):
        """Return a tuple of the Subkey objects in this Vkey.
        """
        return self._subkeys
    
    def get_origins(self):
        """Return a tuple of strings with the origins of the Subkeys in this Vkey.
        """
        return (sk.origin for sk in self.subkeys())
    
    def get_values(self):
        """Return a tuple of strings with the values of the Subkeys in this Vkey.
        """
        return (sk.value for sk in self.subkeys())
    
    def __contains__(self, *subkeys: Union[str, Tuple[str, str]]):
        """Returns True if subkeys that are strings overlap with self.get_values()
        and if subkeys that are string tuples like (`origin`, `value`) have 
        corresponding matches in both self.get_origins() and self.get_values().

        Examples:
            `preproc_0` in vkey => bool
            (`model`, `RF`) in vkey => bool
        """
        _values = self.get_values()
        _sktuples = zip(self.get_origins(), _values)
        for subkey in subkeys:
            if isinstance(subkey, str) and subkey in _values:
                return True
            if isinstance(subkey, tuple) and subkey in _sktuples:
                return True
        return False

    def __add__(self, other: 'Vkey'):
        """Return a new Vkey by combining this Vkey with other, following Subkey
        matching rules. Returns an empty Vkey if there are any Subkey mismatches.
        """
        return Vkey(combine_keys(self.subkeys(), other.subkeys()), self.origin, self.method)

    def __copy__(self):
        """Return a copy of this Vkey (but not its Subkeys).
        """
        return Vkey(self.subkeys(), self.origin, self.method)

    def __deepcopy__(self, memo):
        """Return a copy of this Vkey and its Subkeys.
        """
        return Vkey((sk.__copy__() for sk in self.subkeys()), self.origin, self.method)

    def __len__(self):
        """Return the number of Subkeys in this Vkey.
        """
        return len(self.subkeys())
