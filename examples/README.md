# Example models

These files are written in the `.model` language understood by the
`thermodiff.model` front-end.  Compile any of them with:

```shell
thermodiff-model pr.model -o pr.f90
# or, without the console script
python -m thermodiff.model pr.model -o pr.f90
```

## `pr.model`

A Peng-Robinson style pressure model

$$
f(\mathbf{n}, V, T) =
\frac{R T}{V - b_{\text{mix}}}
- \frac{a_{\text{mix}}}{V\,(V + b_{\text{mix}}) + b_{\text{mix}}\,(V - b_{\text{mix}})}
$$

with

$$
\alpha_i = \left(\frac{T}{T_{c,i}}\right)^2, \quad
a_i = a_{c,i}\,\alpha_i, \quad
a_{\text{mix}} = \sum_i \sum_j n_i n_j \sqrt{a_i a_j}\,(1 - k_{ij}),
\quad
b_{\text{mix}} = \sum_i n_i b_i .
$$

It demonstrates a `use` statement (`R` comes from `yaeos__constants`),
rank-one and rank-two parameter arrays, scalar and indexed intermediate
expressions, and default summations (from `1` to `nc`).

## `molefrac.model`

A mole fraction

$$
x_k = \frac{n_k}{\sum_i n_i}
$$

It demonstrates a function with a **free index** `k`: the generated
subroutine returns `f(:)` and the second compositional derivative is a rank
three array.
