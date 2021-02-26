# sympy2jax

Turn SymPy expressions into parametrized, differentiable, vectorizable, JAX functions.

All SymPy floats become trainable parameters. All SymPy symbols are inputs to the Module.

## Installation

```bash
pip install git+https://github.com/MilesCranmer/sympy2jax.git
```

## Example

```python
import sympy, torch, sympytorch

x = sympy.symbols('x_name')
cosx = 1.0 * sympy.cos(x)
sinx = 2.0 * sympy.sin(x)
mod = sympytorch.SymPyModule(expressions=[cosx, sinx])

x_ = torch.rand(3)
out = mod(x_name=x_)  # out has shape (3, 2)

assert torch.equal(out[:, 0], x_.cos())
assert torch.equal(out[:, 1], 2 * x_.sin())
assert out.requires_grad  # from the two Parameters initialised as 1.0 and 2.0
assert {x.item() for x in mod.parameters()} == {1.0, 2.0}
```
