# How VeridicalFlow works

## Vset

Fundamentally, all `vflow.vset.Vset` objects generate permutations by applying
their modules to the Cartesian product of input parameters. Internally, `Vset`
modules and parameters are wrapped in dictionaries with tuple keys. These keys
contain `vflow.subkey.Subkey` objects, which identify the module or parameter
name, as well as its originating `Vset`. Since inputs must be dictionaries, all
starting parameters must be initialized using `vflow.helpers.init_args`.

When a `Vset` method such as `fit` or `predict` is called with multiple input
arguments, the inputs are first combined left to right by Cartesian product; the
right tuple key is always filtered on any matching `Subkeys` in the left tuple
key, and the two keys' remaining `Subkeys` are concatenated. The values of the
resulting combined dictionary are tuples containing the values of the inputs
concatenated in the order in which they were passed in. (See [`below`](##Subkey)
for more info on `Subkey` matching.) Next, the `Vset` computes the Cartesian
product of the combined input dictionary with `Vset.modules` or `Vset.out`
(depending on the `Vset` method that was called), combining tuple keys in a
similar process to determine which `Vfuncs` to apply to which inputs.

If a `Vset` is initialized with `output_matching=True`, then outputs from this
`Vset` will be matched when used in other `Vsets`. Internally, this means the
new `Subkey` added to the output dictionary by the `Vset` will have an
`output_matching` attribute with value `True`, which is used to reject Cartesian
product combinations when `Subkey` origins do not match or match but their
values differ. A `Vset` should be initialized with `output_matching=True` when
it is used multiple times and outputs from that `Vset` need to be combined
regardless of when it was called, as with a data cleaning `Vset` that is used
first on a training dataset before model fitting and later on a test dataset for
model evaluation; this ensures that the data cleaning procedure used in training
the model matches the data cleaning procedure used when evaluating the model.
(Note that in this example we wouldn't need to initialize the modeling `Vset`
with `output_matching=True`.) On the other hand, if separate calls to the `Vset`
result in entirely independent outputs, such as is usually the case for a `Vset`
that does subsampling, then the `Vset` should be initialized with
`output_matching=False` (the default) to prevent bad matches.

## Subkey

