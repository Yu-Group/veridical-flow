# How vflow Works

## Vset

Fundamentally, all Vsets generate permutations by applying their modules to the cartesian product of parameters. Internally, Vset modules and parameters are wrapped in dictionaries with tuple keys. These keys contain Subkey objects, which identify the module or parameter name, as well as its originating Vset. Since parameters must be dictionaries, all starting parameters must be initialized using [`helpers.init_args`](https://vflow.csinva.io/helpers.html#vflow.helpers.init_args). 

When parameters pass through a Vset, they are combined recursively left to right. Output items are computed by cartesian matching the left and right parameter; the right tuple key is always filtered on any matching Subkeys in the left tuple key, and values are concatenated in tuples. See [`here`](##Subkey) for more info on Subkey matching. The combined parameter dictionary is then combined with the modules dictionary.

If a Vset is initialized with `output_matching=True`, then outputs from this Vset will be matched when used in other Vsets. Internally, this means the modules dictionary's Subkeys have a `output_matching` flag, which is used to reject cartesian outputs when the Subkey origins match but their values differ. `output_matching` is usually set for a Vset that is used multiple times, as with a subsampling or feature extraction Vset.

## Subkey

