""" Incompressible Navier-Stokes equations 
    for backward facing step problem;
    using hybrid method  """

__author__    = "Robert Jan Labeur and Garth N. Wells"
__date__      = "11-11-2011"
__copyright__ = "Copyright (C) 2011 Robert Jan Labeur and Garth N. Wells"
__license__   = "GNU LGPL Version 3"

from dolfin import *

# Code generation options
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["representation"] = "quadrature"

# plot files
outfile_u = File('velo.pvd')
outfile_p = File('pres.pvd')

# polynomial order
k = 2

# advection parameter
chi = .5

# Exact solution inflow boundary
P2 = VectorElement("Lagrange", "triangle", 2)
u_inflow  = Expression(("16.*(1.0 - x[1])*(x[1] - 0.5)", "0.0*x[1]"), element = P2)

# Create mesh
mesh = Rectangle(0.0, 0.0, 15.0, 1.0, 300, 30)

# Define boundaries 
def Lower(x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS

def Upper(x, on_boundary):
    return on_boundary and x[1] >  1. - DOLFIN_EPS

def Step(x, on_boundary):
    return on_boundary and x[0] < DOLFIN_EPS and x[1] - DOLFIN_EPS < .5

def Inflow(x, on_boundary):
    return on_boundary and x[0] < DOLFIN_EPS and x[1] + DOLFIN_EPS > .5

def Corner(x, on_boundary):
    return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Create function spaces
V    = VectorFunctionSpace(mesh, 'DG', k)
Vbar = VectorFunctionSpace(mesh, 'CG', k, restriction='facet')
Q    = FunctionSpace(mesh, 'DG', k)
Qbar = FunctionSpace(mesh, 'CG', k, restriction='facet')

# Create mixed function space
mixed = MixedFunctionSpace([V, Q, Vbar, Qbar])

# Create trial and test functions
v, q, vbar, qbar = TestFunctions(mixed)
u, p, ubar, pbar = TrialFunctions(mixed)

# auxiliary function
Uh = Function(mixed)
Uh.vector()[:]= 0.0*Uh.vector()[:]

# Initial condition
U0 = Function(mixed)
u0, p0, ubar0, pbar0 = split(U0)

# Source term
f = Constant( (0.0, 0.0) )

# Penalty parameters
alpha = Constant(6.0*k*k)
beta = Constant(1.0e-4)

# Mesh related functions
n  = FacetNormal(mesh)
he = CellSize(mesh)

# Dirichlet boundary conditions
bc0 = DirichletBC(mixed.sub(2), Constant((0.,0.)),   Lower)
bc1 = DirichletBC(mixed.sub(2), Constant((0.,0.)),   Upper)
bc2 = DirichletBC(mixed.sub(2), Constant((0.,0.)),   Step)
bc3 = DirichletBC(mixed.sub(2), u_inflow,            Inflow)
bc4 = DirichletBC(mixed.sub(3), Constant(0.0), Corner, "pointwise")
bcs = [bc0, bc1, bc2, bc3, bc4]

# loop over Reynolds-number
for Re in [100., 200., 400., 500., 600., 800.]:

    # viscosity
    nu = 1.0/(1.5*Re)

    # Interface mass flux
    uhat_n       = dot(u, n) - (beta*he/(1.+nu))*(pbar - p)
    uhat0_n      = dot(u0, n) - (beta*he/(1.+nu))*(pbar0 - p0)
    lambda_uhat0 = (uhat0_n - abs(uhat0_n))/2.0

    # Exterior boundary mass flux
    ubar0_n      = dot(ubar0, n)
    lambda_ubar0 = (ubar0_n - abs(ubar0_n))/2.0

    # Momentum fluxes
    sigma_d   = p*Identity(V.cell().d) - 2*nu*sym(grad(u))
    sigma_n_d = pbar*n - 2*nu*sym(grad(u))*n - 2*(nu*alpha/he)*sym(outer(ubar-u,n))*n

    # Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
    def facet_integral(integrand):
        return integrand('-')*dS + integrand('+')*dS + integrand*ds

    # Functionals (solving F = 0)
    F_m_local_a = - chi*inner(outer(u, u0), grad(v))*dx \
                + (1.0 - chi)*dot(grad(u)*u0, v)*dx \
                + chi*facet_integral( uhat0_n*dot(u, v) ) \
                + facet_integral( lambda_uhat0*dot(ubar-u, v) )
    F_m_local_d =  -inner(sigma_d, grad(v))*dx \
                + facet_integral(dot(sigma_n_d, v)) \
                + facet_integral(dot(2*nu*(ubar - u), sym(grad(v))*n))
    F_m_local = F_m_local_a + F_m_local_d

    F_m_global_a = chi*facet_integral( uhat0_n*dot(u, vbar) ) \
                 - (1.0 - chi)*facet_integral( uhat0_n*dot(ubar - u, vbar) )  \
                 + facet_integral( lambda_uhat0*dot(ubar - u, vbar) ) \
                 - chi*dot(ubar0, n)*dot(ubar, vbar)*ds \
                 + lambda_ubar0*dot(ubar, vbar)*ds
    F_m_global_d =   facet_integral(dot(sigma_n_d, vbar))
    F_m_global = F_m_global_a + F_m_global_d

    F_c_local    =   dot(u, grad(q))*dx - facet_integral(uhat_n*q)
    F_c_global   =   facet_integral(uhat_n*qbar) - dot(ubar, n)*qbar*ds
    F = F_m_local + F_m_global + F_c_local + F_c_global

    # Picard iteration
    tol = 1.0e-6
    converged = False
    iter = 0
    u_norm = []
    delta_norm=[]

    while converged == False and iter < 500:
        iter += 1

        print iter

        # update solution
        U0.assign(Uh)

        # Bilinear and linear forms
        a = F
        L = dot(f, v)*dx

        # Solve variational problem
        solve(a == L, Uh, bcs)

        # Compute L2 velocity norm
        uh = Uh.split()[0]
        u_norm.append( sqrt(assemble(dot(uh, uh)*dx, mesh=mesh)) )

        print " u_norm     = ", u_norm[-1]

        # stopping criteria
        if iter < 2:
            converged = False
        else:
            e1 = u_norm[-1]
            e2 = u_norm[-2]
            converged = abs(e1 - e2)/(e1 + e2) < tol
            print " relative increment  =", abs(e1 - e2)/(e1 + e2)

    # plot fields
    outfile_u << Uh.split()[2]
    outfile_p << Uh.split()[3]
