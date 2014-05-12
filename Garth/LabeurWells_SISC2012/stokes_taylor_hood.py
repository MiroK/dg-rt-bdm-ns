""" Solver for time dependent incompressible Navier-Stokes equation
using the GIS method with discontinuous pressure. """

__author__    = "Robert Jan Labeur and Garth N. Wells"
__date__      = "29-10-2010  - 08-09-2011"
__copyright__ = "Copyright (C) 2010-2011 Robert Jan Labeur and Garth N. Wells"
__license__   = "GNU LGPL Version 3"

from dolfin import *

# Code generation options
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["representation"] = "quadrature"

# Error file
error_file = open("errors.out","w")
print >> error_file, 5*'%-25s' % ('polynomial order','cell size (he)', \
    'velocity error','pressure error','divergence error')

# Viscosity
nu = 1.0

# Define boundary
def Gamma(x, on_boundary):  return on_boundary
def Corner(x, on_boundary): return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Exact solution
P7 = VectorElement("Lagrange", "triangle", 8)
P2 = FiniteElement("Lagrange", "triangle", 3)
u_exact = Expression((" x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])", \
                      "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), element = P7)
p_exact = Expression("x[0]*(1.0 - x[0])", element = P2)

# Source term (computed from exact solution)
f = div(p_exact*Identity(2) - 2*nu*sym(grad(u_exact)))

# Mesh sizes
M = [16, 32, 64, 128]

# Loop over polynomial order
for k in [2]:

    error_u = []
    error_p = []
    error_div = []

    for N in M:

        NN = N/k
        print "k=", k, " N=", N, " number of facets along side = ", NN

        # Create mesh
        mesh = UnitSquare(NN, NN)

        # Create function spaces
        V    = VectorFunctionSpace(mesh, 'CG', k)
        Q    = FunctionSpace(mesh, 'CG', k-1)

        # Create mixed function space
        mixed = MixedFunctionSpace([V, Q])

        # Create trial and test functions
        v, q = TestFunctions(mixed)
        u, p = TrialFunctions(mixed)

        # Mesh related functions
        n  = FacetNormal(mesh)
        he = CellSize(mesh)

        # Dirichlet boundary conditions
        bc0 = DirichletBC(mixed.sub(0), u_exact, Gamma)
        bc1 = DirichletBC(mixed.sub(1), Constant(0.), Corner, "pointwise")
        bcs = [bc0,bc1]

        # Momentum fluxes
        sigma_d   = p*Identity(V.cell().d) - 2*nu*sym(grad(u))

        # Functionals (solving F = 0)
        F_m_global =  -inner(sigma_d, grad(v))*dx - dot(f, v)*dx
        F_c_local  =  -div(u)*q*dx
        F = F_m_global + F_c_local

        # Bilinear and linear forms
        a = lhs(F)
        L = rhs(F)

        # Solve variational problem
        Uh = Function(mixed)
        solve(a == L, Uh, bcs)

        # Get solution sub-functions
        uh = Uh.split()[0]
        ph = Uh.split()[1]

        # Compute L2 velocity error
        u_error = uh - u_exact
        error_u.append( sqrt(assemble(dot(u_error, u_error)*dx, mesh=mesh)) )

        # Compute L2 pressure error
        p_error = ph - p_exact
        error_p.append( sqrt(assemble((p_error*p_error)*dx, mesh=mesh)) )

        # Compute divergence error
        error_div.append( sqrt(assemble((div(uh)*div(uh))*dx, mesh=mesh)) )

        print " error_u     = ", error_u[-1]
        print " error_p     = ", error_p[-1]
        print " error_div_u = ", error_div[-1]

    # Write error results to file
    for m, eu, ep, ed in zip(M, error_u, error_p, error_div):
        print >> error_file, 5*'%-25.20f' % (k, 1./(m/k), eu, ep, ed)
