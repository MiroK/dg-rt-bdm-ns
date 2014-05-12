""" Incompressible Stokes equations
for Stokes flow with source test;
using hybrid method with continuous pressure """

__author__    = "Robert Jan Labeur and Garth N. Wells"
__date__      = "11-11-2011"
__copyright__ = "Copyright (C) 2011 Robert Jan Labeur and Garth N. Wells"
__license__   = "GNU LGPL Version 3"

from dolfin import *

# Code generation options
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["representation"] = "quadrature"

# Error file
error_file = open("errors.out", "w")
print >> error_file, 5*'%-25s' % ('polynomial order', 'cell width',
    'velocity error','pressure error','divergence error')

# Viscosity
nu = 1.0

# Define boundary
def Gamma(x, on_boundary): return on_boundary
def Corner(x, on_boundary): return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Exact solution
P7 = VectorElement("Lagrange", "triangle", 8)
P2 = FiniteElement("Lagrange", "triangle", 3)
u_exact = Expression((" x[0]*x[0]*(1.0 - x[0])*(1.0 - x[0])*(2.0*x[1] \
                      - 6.0*x[1]*x[1] + 4.0*x[1]*x[1]*x[1])", \
                      "-x[1]*x[1]*(1.0 - x[1])*(1.0 - x[1])*(2.0*x[0] \
                      - 6.0*x[0]*x[0] + 4.0*x[0]*x[0]*x[0])"), element = P7)
p_exact = Expression("x[0]*(1.0 - x[0])", element = P2)

# Source term (computed from exact solution)
f = div(p_exact*Identity(2) - 2*nu*sym(grad(u_exact)))

# Mesh sizes
M = [16, 32, 64, 128]

# Loop over polynomial order
for k in [1, 2, 3, 4, 5]:

    # Initialise error lists
    error_u = []
    error_p = []
    error_div = []

    for N in M:

        # Number of facets along boundary
        NN = N/k
        print "k=", k, " N=", N, " Facets in each dir=", NN

        # Create mesh
        mesh = UnitSquare(NN, NN)

        # Create function spaces
        V    = VectorFunctionSpace(mesh, 'DG', k)
        Vbar = VectorFunctionSpace(mesh, 'CG', k, restriction='facet')
        Q    = FunctionSpace(mesh, 'CG', k)

        # Create mixed function space
        mixed = MixedFunctionSpace([V, Vbar, Q])

        # Create trial and test functions
        v, vbar, q = TestFunctions(mixed)
        u, ubar, p = TrialFunctions(mixed)

        # Penalty parameters
        alpha = Constant(6*k*k)

        # Mesh related functions
        n  = FacetNormal(mesh)
        he = CellSize(mesh)

        # Dirichlet boundary conditions
        bc0 = DirichletBC(mixed.sub(1), u_exact, Gamma)
        bc1 = DirichletBC(mixed.sub(2), p_exact, Corner, "pointwise")
        bcs = [bc0, bc1]

        # Momentum fluxes
        sigma_d   = p*Identity(V.cell().d) - 2.0*nu*sym(grad(u))
        sigma_n_d = p*n - 2.0*nu*sym(grad(u))*n - (2*nu*alpha/he)*sym(outer(ubar - u,n))*n

        # Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
        def facet_integral(integrand):
            return integrand('-')*dS + integrand('+')*dS + integrand*ds

        # Problem defintion (F = 0)
        F_m_local  =  -inner(sigma_d, grad(v))*dx + facet_integral(dot(sigma_n_d, v)) \
                     + facet_integral(dot((ubar - u), 2.0*nu*sym(grad(v))*n)) - dot(f, v)*dx
        F_m_global =   facet_integral(dot(sigma_n_d, vbar))
        F_c        =   dot(u, grad(q))*dx - facet_integral(dot(ubar, n)*q)
        F = F_m_local + F_m_global + F_c

        # Bilinear and linear forms
        a, L = lhs(F), rhs(F)

        # Solve variational problem
        Uh = Function(mixed)
        solve(a == L, Uh, bcs)

        # Get solution sub-functions
        uh = Uh.split()[0]
        ph = Uh.split()[2]

        # Compute L2 velocity error
        u_error = uh - u_exact
        error_u.append( sqrt(abs(assemble(dot(u_error, u_error)*dx, mesh=mesh))) )

        # Compute L2 pressure error
        p_error = ph - p_exact
        error_p.append( sqrt(abs(assemble(p_error*p_error*dx, mesh=mesh))) )

        # Compute L2-norm divergence
        error_div.append( sqrt(abs(assemble(div(uh)*div(uh)*dx, mesh=mesh))) )

        print " error_u     = ", error_u[-1]
        print " error_p     = ", error_p[-1]
        print " error_div_u = ", error_div[-1]

    # Write results to file
    for m, eu, ep, ed in zip(M, error_u, error_p, error_div):
        print >> error_file, 5*'%-25.20f' % (k, 1./(m/k), eu, ep, ed)
