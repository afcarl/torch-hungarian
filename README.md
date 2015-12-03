# Torch wrapper of  C Implementation of the Hungarian Method #

[C Implementation of the Hungarian Method](http://www2.informatik.uni-freiburg.de/~stachnis/misc.html) 

## Usage

```lua
> require 'hungarian'
> x = torch.rand(4, 5)
> values, indices = hungarian.solve(x)
```
