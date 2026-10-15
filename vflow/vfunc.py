"""A perturbation that can be used as a step in a pipeline
"""

from typing import Callable

import ray


def _check_vfunc(vfunc):
    if not (hasattr(vfunc, "fit") or callable(vfunc)):
        raise TypeError("vfunc must be an object with a fit method or a callable")


def _method_or_self(vfunc, method: str):
    """`vfunc.<method>` (e.g. an estimator's fit) if it has one, else `vfunc`
    itself (a plain function)."""
    return getattr(vfunc, method, vfunc)


class Vfunc:
    """Vfunc is basically a function along with a name attribute.
    It may support a "fit" function, but may also just have a "transform" function.
    If none of these is supported, it need only be a function
    """

    def __init__(self, name: str = "", vfunc=lambda x: x):
        _check_vfunc(vfunc)
        self.name = name
        self.vfunc = vfunc

    def fit(self, *args, **kwargs):
        """Calls the vfunc's fit method, or the vfunc itself if it has none"""
        return _method_or_self(self.vfunc, "fit")(*args, **kwargs)

    def transform(self, *args, **kwargs):
        """Calls the vfunc's transform method, or the vfunc itself if it has none"""
        return _method_or_self(self.vfunc, "transform")(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Same as fit"""
        return self.fit(*args, **kwargs)


@ray.remote
def _remote_fun(vfunc, *args, **kwargs):
    return vfunc(*args, **kwargs)


class AsyncVfunc:
    """An asynchronous version of the Vfunc class."""

    def __init__(self, name: str = "", vfunc=lambda x: x):
        self.name = name
        if isinstance(vfunc, Vfunc):
            vfunc = vfunc.vfunc
        _check_vfunc(vfunc)
        self.vfunc = vfunc

    def fit(self, *args, **kwargs):
        """Like Vfunc.fit, run as a Ray task; returns a Ray reference"""
        return _remote_fun.remote(_method_or_self(self.vfunc, "fit"), *args, **kwargs)

    def transform(self, *args, **kwargs):
        """Like Vfunc.transform, run as a Ray task; returns a Ray reference"""
        return _remote_fun.remote(
            _method_or_self(self.vfunc, "transform"), *args, **kwargs
        )

    def __call__(self, *args, **kwargs):
        """Same as fit"""
        return self.fit(*args, **kwargs)


class VfuncPromise:
    """A Vfunc promise used for lazy evaluation."""

    def __init__(self, vfunc: Callable, *args):
        self.vfunc = vfunc
        self.args = args
        self.called = False
        self.value = None

    def __call__(self):
        """Evaluates the promise the first time; returns the stored value after"""
        if self.called:
            return self.value
        tmp_args = []
        for arg in self.args:
            while isinstance(arg, VfuncPromise):
                arg = arg()
            tmp_args.append(arg)
        while isinstance(self.vfunc, VfuncPromise):
            self.vfunc = self.vfunc()
        # an upstream lazy async Vset leaves Ray references; a remote call
        # resolves them itself, but a plain function must get the values
        if not isinstance(getattr(self.vfunc, "__self__", None), AsyncVfunc):
            tmp_args = [
                ray.get(a) if isinstance(a, ray.ObjectRef) else a for a in tmp_args
            ]
        self.value = self.vfunc(*tmp_args)
        self.called = True
        return self.value

    def _get_value(self):
        if isinstance(self(), ray.ObjectRef):
            self.value = ray.get(self.value)
        return self.value

    def transform(self, *args):
        """This function transforms its input in some way"""
        return self._get_value().transform(*args)

    def predict(self, *args):
        """This function calls predict on its inputs"""
        return self._get_value().predict(*args)

    def predict_proba(self, *args):
        """This function calls predict_proba on its inputs"""
        return self._get_value().predict_proba(*args)

    def __repr__(self):
        if self.called:
            return f"Fulfilled VfuncPromise({self.value})"
        return f"Unfulfilled VfuncPromise(func={self.vfunc}, args={self.args})"
