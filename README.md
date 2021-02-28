# sympy2jax

Turn SymPy expressions into parametrized, differentiable, vectorizable, JAX functions.

All SymPy floats become trainable input parameters.
SymPy symbols become columns of a passed matrix.

## Installation

```bash
pip install git+https://github.com/MilesCranmer/sympy2jax.git
```

## Example

```python
import sympy
from sympy import symbols
import jax
import jax.numpy as jnp
from jax import random
from sympy2jax import sympy2jax
```

Let's create an expression in SymPy:
```python
x, y = symbols('x y')
expression = 1.0 * sympy.cos(x) + 3.2 * y
```
Let's get the JAX version. We pass the equation, and
the symbols required.
```python
f, params = sympy2jax(expression, [x, y])
```
The order you supply the symbols is the same order
you should supply the features when calling
the function `f` (shape `[nrows, nfeatures]`).
In this case, features=2 for x and y.
The `params` in this case will be
`jnp.array([1.0, 3.2])`. You pass these parameters
when calling the function, which will let you change them
and take gradients.

Let's generate some JAX data to pass:
```python
key = random.PRNGKey(0)
X = random.normal(key, (10, 2))
```

We can call the function with:
```python
f(X, params)

#> DeviceArray([-2.6080756 ,  0.72633684, -6.7557726 , -0.2963162 ,
#                6.6014843 ,  5.032483  , -0.810931  ,  4.2520013 ,
#                3.5427954 , -2.7479894 ], dtype=float32)
```

We can take gradients with respect
to the parameters for each row with JAX
gradient parameters now:
```python
jac_f = jax.jacobian(f, argnums=1)
jac_f(X, params)

#> DeviceArray([[ 0.49364874, -0.9692889 ],
#               [ 0.8283714 , -0.0318858 ],
#               [-0.7447336 , -1.8784496 ],
#               [ 0.70755106, -0.3137085 ],
#               [ 0.944834  ,  1.767703  ],
#               [ 0.51673377,  1.4111717 ],
#               [ 0.87347716, -0.52637756],
#               [ 0.8760679 ,  1.0549792 ],
#               [ 0.9961824 ,  0.79581654],
#               [-0.88465923, -0.5822907 ]], dtype=float32)
```

We can also JIT-compile our function:
```python
compiled_f = jax.jit(f)
compiled_f(X, params)

#> DeviceArray([-2.6080756 ,  0.72633684, -6.7557726 , -0.2963162 ,
#                6.6014843 ,  5.032483  , -0.810931  ,  4.2520013 ,
#                3.5427954 , -2.7479894 ], dtype=float32)
```
