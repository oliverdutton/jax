
import jax
from jax.extend.core import Jaxpr, Var, JaxprEqn, ClosedJaxpr, Literal
from collections import Counter
import hashlib

def _hash_eqn(eqn):
    """Create a hashable key for a JaxprEqn."""
    # This is a very basic hash, might need to be more robust
    # For example, it doesn't handle nested jaxprs well.
    invars_repr = ""
    for var in eqn.invars:
        if isinstance(var, Literal):
            invars_repr += str(var.val)
        else:
            invars_repr += str(var)
    params_repr = "".join(map(str, eqn.params.values()))
    return hashlib.md5((str(eqn.primitive) + invars_repr + params_repr).encode()).hexdigest()

def cse_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
    """Performs common subexpression elimination on a Jaxpr."""
    new_eqns = []
    cse_cache = {}
    substitutions = {}

    for eqn in jaxpr.eqns:
        # Recursively apply CSE to nested jaxprs
        new_params = {}
        for k, v in eqn.params.items():
            if isinstance(v, Jaxpr):
                new_params[k] = cse_jaxpr(v)
            elif isinstance(v, ClosedJaxpr):
                new_params[k] = ClosedJaxpr(cse_jaxpr(v.jaxpr), v.consts)
            else:
                new_params[k] = v

        # Substitute inputs to the current equation
        new_invars = []
        for var in eqn.invars:
            if isinstance(var, Var):
                new_invars.append(substitutions.get(var, var))
            else:
                new_invars.append(var)
        new_eqn = eqn.replace(invars=new_invars, params=new_params)

        eqn_hash = _hash_eqn(new_eqn)

        if eqn_hash in cse_cache:
            # If we've seen this exact computation before, substitute the output
            # of this equation with the output of the cached equation.
            for out_var, cached_out_var in zip(new_eqn.outvars, cse_cache[eqn_hash].outvars):
                substitutions[out_var] = cached_out_var
        else:
            # This is a new computation, add it to our list of equations
            # and cache it.
            cse_cache[eqn_hash] = new_eqn
            new_eqns.append(new_eqn)

    # Substitute the outputs of the jaxpr
    new_outvars = [substitutions.get(var, var) for var in jaxpr.outvars]

    return Jaxpr(
        constvars=jaxpr.constvars,
        invars=jaxpr.invars,
        outvars=new_outvars,
        eqns=new_eqns,
    )
