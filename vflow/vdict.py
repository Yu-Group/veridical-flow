"""Vdict key tuple wrapper class
"""
from typing import Tuple, Union

class Vdict:

    def __init__(self, _dict: dict = {}):
        self._dict = _dict

    def to_pandas(self, copy=False):
        """Return a pandas.DataFrame representation of the Vdict.
        """

    def __getitem__(self, *subkeys: Union[str, Tuple[str, str]], copy=False):
        """Return a new Vdict with a subset of the items in self by filtering keys
        based on subkey values. If copy=True, then make a deep copy of Vkeys and values.

        Examples:
            preds[`preproc_0`, `RF`] => Vdict with all items that have subkey
                                        with value `preproc_0` and another with `RF`
            `preproc_0` in preds => bool
            (`model`, `RF`) in preds => bool
        """
        out = Vdict()
        for vkey, value in self._dict.items():
            if subkeys in vkey:
                if copy:
                    out[vkey.__deepcopy__()] = value
                else:
                    out[vkey] = value
        return out
