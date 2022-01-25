# How VeridicalFlow works

VeridicalFlow provides two main abstractions: `vflow.vset.Vset`,
`vflow.vfunc.Vfunc`, `vflow.subkey.Subkey`. `Vfunc`s run arbitrary computations
on inputs. `Vset`s collect related `Vfunc`s and determine which `Vfunc`s to
apply to which inputs and how to do so. `Subkey`s help associate outputs with
the `Vfunc` that produced them and the `Vset` in which the `Vfunc` was
collected. Below we describe these three abstractions in more detail.

## Vset

Fundamentally, all `vflow.vset.Vset` objects generate permutations by applying
their modules to the Cartesian product of input parameters. Internally, `Vset`
modules and parameters are wrapped in dictionaries with tuple keys. These tuples
contain `vflow.subkey.Subkey` objects, which identify the `vflow.vfunc.Vfunc` or
parameter name (the `Subkey` "value"), as well as its originating `Vset` (the
`Subkey` "origin"). Since inputs must be dictionaries with a certain format, all
raw inputs must be initialized using `vflow.helpers.init_args`.

When a `Vset` method such as `fit` or `predict` is called with multiple input
arguments, the inputs are first combined left to right by Cartesian product; the
right `Subkey` tuple is always filtered on any matching `Subkeys` in the left
`Subkey` tuple, and the two keys' remaining `Subkeys` are concatenated. The
values of the resulting combined dictionary are tuples containing the values of
the inputs concatenated in the order in which they were passed in. (See
[`below`](##Subkey) for more info on `Subkey` matching.) Next, the `Vset`
computes the Cartesian product of the combined input dictionary with
`Vset.modules` or `Vset.out` (depending on the `Vset` method that was called),
combining `Subkey` tuples in a similar process to determine which `Vfuncs` to
apply to which inputs.

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
(In contrast, note that in this example we wouldn't need to initialize the
modeling `Vset` with `output_matching=True`.) On the other hand, if separate
calls to the `Vset` result in entirely independent outputs, such as is usually
the case for a `Vset` that does subsampling, then the `Vset` should be
initialized with `output_matching=False` (the default) to prevent bad matches
(e.g., unnecessary matching on subsamples of two different datasets).

## Vfunc

When a `Vset` is initialized, the items in the `modules` arg are wrapped in
`Vfunc` objects, which are like named functions that can optionally support a
`fit` or `transform` method. If a `Vset` is initialized with `is_async=True`,
then modules are wrapped in `AsyncModule` objects, which compute asynchronously
using `ray`. If a `Vset` is initialized with `lazy=True`, `Vset` output
dictionary values are wrapped in `VfuncPromise` objects, which are lazily
evaluated. At the moment, `VfuncPromise` objects are only resolved if passed to
a `Vset` with `lazy=False` downstream or if called manually by the user.

## Subkey

By default, `vflow.subkey.Subkey` objects are non-matching, meaning that `vflow`
won't bother to look for a matching `Subkey` when combining data or deciding
which `Vfunc` to apply to which inputs. As described above, `Vset` inputs are
combined from left to right two-at-a-time, and if every `Subkey` in the keys of
both input dictionaries is non-matching, then the result is the full Cartesian
product of the two dictionaries. (Note, however, that later inputs where
matching is required may invalidate some entries in that Cartesian product.)
There is one main way for a `Subkey` to be matching: it was created by a `Vset`
that was initialized with `output_matching=True`.

When `vflow` combines two dictionaries it previously created, if one of the
entries in a given `Subkey` tuple key is matching then `vflow` tries to find a
match in the tuple keys of the other dictionary. If the other dictionary's
`Subkey`s are non-matching, or if there is a matching `Subkey` with the same
origin and same value, then the keys are combined and their values are either
combined (when combining inputs) or a `Vfunc` is applied to matching input (when
using previously combined inputs). For example, say we have a data cleaning
`Vset` named "cleaning" that was initialized with `output_matching=True` and has
a `Vfunc` named "median impute" and another named "mean impute", and say this
`Vset` was applied to a training set and later to a test set. When the `Vset` is
applied to the training data, the output dictionary of cleaned datasets has some
tuple keys containing a `Subkey` with origin "cleaning" and value "median
impute", and that `Subkey` is carried through when the cleaned data is passed to
a modeling `Vset`. When we want to produce test predictions from the modeling
`Vset`, `vflow` decides which trained models to apply to which version of the
cleaned test data based on the `Subkey`s created by the data cleaning `Vset`. We
have a positive match if the tuple keys each have a `Subkey` with origin
"cleaning" and value "median impute". On the other hand, if one of the `Subkey`s
instead had value "mean impute", then the two keys mismatch and we proceed to
check the next dictionary key without applying the trained model to the test
data.
