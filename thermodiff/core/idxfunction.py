import sympy as sp


def IdxFunction(name):
    return type(
        name,
        (sp.Function,),
        {
            "_eval_derivative": lambda self, s: sp.Derivative(
                self, s, evaluate=False
            ),
            "__doc__": f"Simbolic function {name} with lazy derivative",
        },
    )
