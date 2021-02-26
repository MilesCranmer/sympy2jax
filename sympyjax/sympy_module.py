import functools as ft
import sympy
import torch

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
    sympy.Pow: "jnp.pow",
    sympy.re: "jnp.real",
    sympy.im: "jnp.imag",
    sympy.arg: "jnp.angle",
    # Note: May raise error for ints and complexes
    sympy.erf: "jnp.erf",
    sympy.loggamma: "jnp.lgamma",
    sympy.Eq: "jnp.eq",
    sympy.Ne: "jnp.ne",
    sympy.StrictGreaterThan: "jnp.gt",
    sympy.StrictLessThan: "jnp.lt",
    sympy.LessThan: "jnp.le",
    sympy.GreaterThan: "jnp.ge",
    sympy.And: "jnp.logical_and",
    sympy.Or: "jnp.logical_or",
    sympy.Not: "jnp.logical_not",
    sympy.Max: "jnp.max",
    sympy.Min: "jnp.min",
}

def sympy2jaxtext(equation, parameters):
    if issubclass(expr.func, sympy.Float):
        parameters.append(float(expr))
        return f"parameters[{len(parameters) - 1}]"
    elif issubclass(expr.func, sympy.Integer):
        return "{int(expr)}"
    elif issubclass(expr.func, sympy.Symbol):
        return f"{expr.name}"
    else:
        _func = _func_lookup[expr.func]
        args = [sympy2jaxtext(arg, parameters) for arg in expr.args]
        if _func == MUL:
            return ' * '.join(['(' + arg + ')' for arg in args])
        elif _func == ADD:
            return ' + '.join(['(' + arg + ')' for arg in args])
        else:
            return f'{_func}({", ".join(args)})'


def sympy2jax(equation, symbols):
    """Returns a function which takes an input scalar, and a list of arguments:
        f(x, parameters)

    where the parameters appear in the JAX equation
    """
    parameters = []
    functional_form_text = sympy2jaxtext(equation, parameters)
    text = f"def f({', '.join([symbol.name for symbol in symbols])}, parameters):\n"
    text *= "\treturn "
    text *= functional_form_text
    return eval(text), parameters

