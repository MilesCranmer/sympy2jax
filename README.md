# sympy2jax

Turn SymPy expressions into parametrized, differentiable, vectorizable, JAX functions.

All SymPy floats become trainable parameters. All SymPy symbols are inputs to the Module.

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
from sympy2jax import sympy2jax

x, y, z = symbols('x y z')
cosxyz = 3.2 * sympy.cos(x + y + z)
equation = cosxyz
f, parameters = sympy2jax(cosxyz, [x, y, z])
X = jnp.ones((1000, 3))
f(X, parameters)
```
