from functools import partial
from dolfin import *
import ascot

# Inner product on RT_k space
def h1_div(v, u, n, h):
    m = inner(u, v)*dx + inner(grad(u), grad(v))*dx# +\
        #inner(jump(outer(u, n)), jump(outer(v, n)))*dS +\
        #inner(outer(u, n), outer(v, n))*ds
    return m

# Inner product in DG_k space
def l2(q, p):
    n = inner(p, q)*dx
    return n

# Cross term in whose stability we investigate
b = lambda v, q : inner(div(v), q)*dx

# Create sequence of spaces W^k = RT_k x DG_k
k = 1
betas = []
for N in [2, 4, 6, 8, 10][:3]:
    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'BDM', k+1)
    Q = FunctionSpace(mesh, 'DG', k)
    W = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h1_div = partial(h1_div, n=n, h=h)
    
    betas.append(ascot.compute_brezzi_infsup(b, (h1_div, l2), W))

collection = ascot.InfSupCollection(betas)
result = ascot.StabilityResult(collection)

print collection
print result
