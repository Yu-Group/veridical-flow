"""A perturbation that can be used as a step in a pipeline
"""

import ray


class Vfunc:
    """Vfunc is basically a function along with a name attribute.
    It may support a "fit" function, but may also just have a "transform" function.
    If none of these is supported, it need only be a function
    """

    def __init__(self, name: str = '', vfunc=lambda x: x):
        assert hasattr(vfunc, 'fit') or callable(vfunc), \
            'vfunc must be an object with a fit method or a callable'
        self.name = name
        self.vfunc = vfunc

    def fit(self, *args, **kwargs):
        """This function fits params for this vfunc
        """
        if hasattr(self.vfunc, 'fit'):
            return self.vfunc.fit(*args, **kwargs)
        return self.vfunc(*args, **kwargs)

    def transform(self, *args, **kwargs):
        """This function transforms its input in some way
        """
        if hasattr(self.vfunc, 'transform'):
            return self.vfunc.transform(*args, **kwargs)
        return self.vfunc(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """This should decide what to call
        """
        return self.fit(*args, **kwargs)


@ray.remote
def _remote_fun(vfunc, *args, **kwargs):
    return vfunc(*args, **kwargs)


class AsyncVfunc:
    """An asynchronous version of the Vfunc class.
    """

    def __init__(self, name: str = '', vfunc=lambda x: x):
        self.name = name
        if isinstance(vfunc, Vfunc):
            self.vfunc = vfunc.vfunc
        else:
            assert hasattr(vfunc, 'fit') or callable(vfunc), \
                'vfunc must be an object with a fit method or a callable'
            self.vfunc = vfunc

    def fit(self, *args, **kwargs):
        """This function fits params for this vfunc
        """
        if hasattr(self.vfunc, 'fit'):
            return _remote_fun.remote(self.vfunc.fit, *args, **kwargs)
        return _remote_fun.remote(self.vfunc, *args, **kwargs)

    def transform(self, *args, **kwargs):
        """This function transforms its input in some way
        """
        if hasattr(self.vfunc, 'transform'):
            return _remote_fun.remote(self.vfunc.transform, *args, **kwargs)
        return _remote_fun.remote(self.vfunc, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """This should decide what to call
        """
        return self.fit(*args, **kwargs)


class VfuncPromise:
    """A Vfunc promise used for lazy evaluation.
    """

    def __init__(self, vfunc: callable, *args):
        self.vfunc = vfunc
        self.args = args
        self.called = False
        self.value = None

    def __call__(self):
        """This should decide what to call
        """
        if self.called:
            return self.value
        tmp_args = []
        for i, arg in enumerate(self.args):
            tmp_args.append(arg)
            while isinstance(tmp_args[i], VfuncPromise):
                tmp_args[i] = tmp_args[i]()
        while isinstance(self.vfunc, VfuncPromise):
            self.vfunc = self.vfunc()
        self.value = self.vfunc(*tmp_args)
        self.called = True
        return self.value

    def _get_value(self):
        if isinstance(self(), ray.ObjectRef):
            self.value = ray.get(self.value)
        return self.value

    def transform(self, *args):
        """This function transforms its input in some way
        """
        return self._get_value().transform(*args)

    def predict(self, *args):
        """This function calls predict on its inputs
        """
        return self._get_value().predict(*args)

    def predict_proba(self, *args):
        """This function calls predict_proba on its inputs
        """
        return self._get_value().predict_proba(*args)

    def __repr__(self):
        if self.called:
            return f'Fulfilled VfuncPromise({self.value})'
        return f'Unfulfilled VfuncPromise(func={self.vfunc}, args={self.args})'
