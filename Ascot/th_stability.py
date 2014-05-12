'''This script is my tutorial on ASCoT.'''

from dolfin import *
import ascot

# Inner product on V ([CG_k]^d)
def h1(v, u):
    m = inner(u, v)*dx + inner(grad(u), grad(v))*dx
    return m

# Inner product in Q (DG_k)
def l2(q, p):
    n = inner(p, q)*dx
    return n

# Cross term in whose stability we investigate
b = lambda v, q : inner(div(v), q)*dx

# Create sequence of spaces W^k = RT_k x DG_k
k = 2
betas = [] # Stability constants
for N in [2, 4, 6, 8, 10, 12, 14, 16]:
    mesh = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(mesh, 'CG', k)
    Q = FunctionSpace(mesh, 'CG', k-1)
    W = MixedFunctionSpace([V, Q])
    
    # Setting bcs leads to only reduced stable system
    # This is alleviated by changing to norm to H10
    #bc = DirichletBC(W.sub(0), Constant((0., 0.)), DomainBoundary())
    bc = None

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    betas.append(ascot.compute_brezzi_infsup(b, (h1, l2), W, bc=bc))

collection = ascot.InfSupCollection(betas)
result = ascot.StabilityResult(collection)

print collection
print result



