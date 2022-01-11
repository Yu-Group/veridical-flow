"""A perturbation that can be used as a step in a pipeline
"""

import ray


class Vfunc:
    """Vfunc is basically a function along with a name attribute.
    It may support a "fit" function, but may also just have a "transform" function.
    If none of these is supported, it need only be a function
    """

    def __init__(self, name: str = '', module=lambda x: x):
        assert hasattr(module, 'fit') or callable(module), \
            'module must be an object with a fit method or a callable'
        self.name = name
        self.module = module

    def fit(self, *args, **kwargs):
        """This function fits params for this module
        """
        if hasattr(self.module, 'fit'):
            return self.module.fit(*args, **kwargs)
        return self.module(*args, **kwargs)

    def transform(self, *args, **kwargs):
        """This function transforms its input in some way
        """
        if hasattr(self.module, 'transform'):
            return self.module.transform(*args, **kwargs)
        return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """This should decide what to call
        """
        return self.fit(*args, **kwargs)


@ray.remote
def _remote_fun(module, *args, **kwargs):
    return module(*args, **kwargs)


class AsyncModule:
    """An asynchronous version of the Vfunc class.
    """

    def __init__(self, name: str = '', module=lambda x: x):
        self.name = name
        if isinstance(module, Vfunc):
            self.module = module.module
        else:
            assert hasattr(module, 'fit') or callable(module), \
                'module must be an object with a fit method or a callable'
            self.module = module

    def fit(self, *args, **kwargs):
        """This function fits params for this module
        """
        if hasattr(self.module, 'fit'):
            return _remote_fun.remote(self.module.fit, *args, **kwargs)
        return _remote_fun.remote(self.module, *args, **kwargs)

    def transform(self, *args, **kwargs):
        """This function transforms its input in some way
        """
        if hasattr(self.module, 'transform'):
            return _remote_fun.remote(self.module.transform, *args, **kwargs)
        return _remote_fun.remote(self.module, *args, **kwargs)

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
