import functools as ft
import sympy
import jax
from jax import numpy as jnp
from jax.scipy import special as jsp

def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)
    return fn_

# Special since need to reduce arguments.
MUL = 0
ADD = 1

_func_lookup = {
    sympy.Mul: MUL,
    sympy.Add: ADD,
    sympy.div: "jnp.div",
    sympy.Abs: "jnp.abs",
    sympy.sign: "jnp.sign",
    # Note: May raise error for ints.
    sympy.ceiling: "jnp.ceil",
    sympy.floor: "jnp.floor",
    sympy.log: "jnp.log",
    sympy.exp: "jnp.exp",
    sympy.sqrt: "jnp.sqrt",
    sympy.cos: "jnp.cos",
    sympy.acos: "jnp.acos",
    sympy.sin: "jnp.sin",
    sympy.asin: "jnp.asin",
    sympy.tan: "jnp.tan",
    sympy.atan: "jnp.atan",
    sympy.atan2: "jnp.atan2",
    # Note: Also may give NaN for complex results.
    sympy.cosh: "jnp.cosh",
    sympy.acosh: "jnp.acosh",
    sympy.sinh: "jnp.sinh",
    sympy.asinh: "jnp.asinh",
    sympy.tanh: "jnp.tanh",
    sympy.atanh: "jnp.atanh",
    sympy.Pow: "jnp.power",
    sympy.re: "jnp.real",
    sympy.im: "jnp.imag",
    sympy.arg: "jnp.angle",
    # Note: May raise error for ints and complexes
    sympy.erf: "jsp.erf",
    sympy.LessThan: "jnp.le",
    sympy.GreaterThan: "jnp.ge",
    sympy.And: "jnp.logical_and",
    sympy.Or: "jnp.logical_or",
    sympy.Not: "jnp.logical_not",
    sympy.Max: "jnp.max",
    sympy.Min: "jnp.min",
}

def sympy2jaxtext(expr, parameters, symbols_in):
    if issubclass(expr.func, sympy.Float):
        parameters.append(float(expr))
        return f"parameters[{len(parameters) - 1}]"
    elif issubclass(expr.func, sympy.Integer):
        return "{int(expr)}"
    elif issubclass(expr.func, sympy.Symbol):
        return f"X[:, {[i for i in range(len(symbols_in)) if symbols_in[i] == expr][0]}]"
    else:
        _func = _func_lookup[expr.func]
        args = [sympy2jaxtext(arg, parameters, symbols_in) for arg in expr.args]
        if _func == MUL:
            return ' * '.join(['(' + arg + ')' for arg in args])
        elif _func == ADD:
            return ' + '.join(['(' + arg + ')' for arg in args])
        else:
            return f'{_func}({", ".join(args)})'

def sympy2jax(equation, symbols_in):
    """Returns a function which takes an input matrix, and a list of arguments:
            f(X, parameters)
    where the parameters appear in the JAX equation.
    """
    parameters = []
    functional_form_text = sympy2jaxtext(equation, parameters, symbols_in)
    text = f"def f(X, parameters):\n"
    text += "    return "
    text += functional_form_text
    ldict = {}
    exec(text, globals(), ldict)
    return ldict['f'], jnp.array(parameters)

