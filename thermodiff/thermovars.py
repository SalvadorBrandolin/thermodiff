import sympy as sp

# Variable dummy Nc (numero de componentes)
nc = sp.symbols("N_c", integer=True)

# Indices
# Para derivadas de n, el usuario no usa estos índices directamente,
i = sp.symbols("i", cls=sp.Idx)
j = sp.symbols("j", cls=sp.Idx)

# Indices para variables y cosas
k = sp.symbols("k", cls=sp.Idx)
l = sp.symbols("l", cls=sp.Idx)
m = sp.symbols("m", cls=sp.Idx)

# Thermo variables
n = sp.IndexedBase("n", shape=(nc,))
T = sp.symbols("T")
V = sp.symbols("V")
