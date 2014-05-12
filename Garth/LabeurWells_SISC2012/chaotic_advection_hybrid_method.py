""" Incompressible Navier-Stokes equations
for chaotic advection test;
using the hybrid method """

__author__    = "Robert Jan Labeur and Garth N. Wells"
__date__      = "11-11-2011"
__copyright__ = "Copyright (C) 2011 Robert Jan Labeur and Garth N. Wells"
__license__   = "GNU LGPL Version 3"

from dolfin import *
import numpy
import pickle

# Code generation options
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["representation"] = "quadrature"

# Random source term generator
class Source(Expression):
    def __init__(self):
        numpy.random.seed(2 + MPI.process_number())

    def eval(self, values, x):
        values[0]= numpy.random.uniform(-1.0, 1.0)
        values[1]= numpy.random.uniform(1.0, 1.0)

    def value_shape(self):
        return (2,)

# Mesh
mesh = UnitSquare(32, 32)

# Define boundaries
def LeftRight(x, on_boundary):
    return x[0] < DOLFIN_EPS or (1.0 - x[0]) < DOLFIN_EPS
def UpperLower(x, on_boundary):
    return x[1] < DOLFIN_EPS or (1.0 - x[1]) < DOLFIN_EPS
def Corner(x, on_boundary):
    return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Output files
outfile_u = File('./output/velocity.pvd')
outfile_p = File('./output/pressure.pvd')

# Time stepping
dt = Constant(2.0e-1)                       # time step size
theta0 = 1.0                                # theta for first step
theta1 = 0.5                                # theta for remainder
theta_change_step = 5                       # change theta after 5 steps
num_steps  = 1000 + theta_change_step       # total number of steps
theta = Constant(theta0, cell=triangle)     # assign theta

# Polynomial order
k = 1

# Penalty parameters
alpha = Constant(6.0*k*k)
beta  = 1.0e-6

# Viscosity
nu = Constant(1.0/1.0e5)

# Advective (0) or conservative (1) form advection operator
chi = 0.5

# Function spaces
V     = VectorFunctionSpace(mesh, 'DG', k)
Vbar  = VectorFunctionSpace(mesh, 'CG', k, restriction='facet')
Q     = FunctionSpace(mesh, 'DG', k)
Qbar  = FunctionSpace(mesh, 'CG', k, restriction='facet')

# Mixed finite element space
mixed = MixedFunctionSpace([V, Vbar, Q, Qbar])
v, vbar, q, qbar = TestFunctions(mixed)

# Trial functions
U = TrialFunction(mixed)
u, ubar, p, pbar = split(U)

# Solution at previous time step
U0 = Function(mixed)
u0, ubar0, p0, pbar0 = split(U0)

# Source term
P1 = VectorFunctionSpace(mesh, 'CG', 1)
f  = Function(P1)

# Mesh related functions
n  = FacetNormal(mesh)
he = CellSize(mesh)

# Mid-point values
u_theta    = (1.0-theta)*u0    + theta*u
ubar_theta = (1.0-theta)*ubar0 + theta*ubar
p_theta    = (1.0-theta)*p0    + theta*p
pbar_theta = (1.0-theta)*pbar0 + theta*pbar

# Interface mass flux terms
uhat_n       = dot(u,  n) - beta*(pbar - p)
uhat0_n      = dot(u0, n) - beta*(pbar0 - p0)
lambda_uhat0 = (uhat0_n - abs(uhat0_n))/2.0

# Exterior boundary mass flux terms
ubar0_n  = dot(ubar0, n)
lambda_ubar0 = (ubar0_n - abs(ubar0_n))/2.0

# Deformation rate
D_theta = (1.0-theta)*sym(grad(u0)) + theta*sym(grad(u))

# Diffusive momentum fluxes
sigma_d   = p_theta*Identity(V.cell().d) - 2.0*nu*D_theta
sigma_n_d = pbar_theta*n - 2.0*nu*D_theta*n \
           - (2*nu*alpha/he)*sym(outer(ubar_theta - u_theta,n))*n

# Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
def facet_integral(integrand):
    return integrand('-')*dS + integrand('+')*dS + integrand*ds

# Define functionals
F_m_local_d = dot((u - u0)/dt, v)*dx \
             - inner(sigma_d, grad(v))*dx \
             + facet_integral( dot(sigma_n_d, v) ) \
             + facet_integral( dot(ubar_theta - u_theta, 2*nu*sym(grad(v))*n) ) \
             - dot(f, v)*dx
F_m_local_a = -chi*inner(outer(u_theta, u0), grad(v))*dx \
             + (1.0 - chi)*dot(grad(u_theta)*u0, v)*dx \
             + chi*facet_integral( uhat0_n*dot(u_theta, v) ) \
             + facet_integral( lambda_uhat0*dot(ubar_theta - u_theta, v) )
F_m_local = F_m_local_d + F_m_local_a

F_m_global_d =  facet_integral( dot(sigma_n_d, vbar) )
F_m_global_a = chi*facet_integral( uhat0_n*dot(u_theta, vbar) ) \
             - (1.0 - chi)*facet_integral( uhat0_n*dot(ubar_theta - u_theta, vbar) )  \
             + facet_integral( lambda_uhat0*dot(ubar_theta - u_theta, vbar) ) \
             - chi*dot(ubar0, n)*dot(ubar_theta, vbar)*ds \
             + lambda_ubar0*dot(ubar_theta, vbar)*ds
F_m_global = F_m_global_d + F_m_global_a

F_c_local  = dot(u, grad(q))*dx - facet_integral(uhat_n*q)
F_c_global = facet_integral(uhat_n*qbar) - dot(ubar, n)*qbar*ds

# Sum all terms
F = F_m_local + F_m_global + F_c_local + F_c_global

# Bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Create intitial source term and interpolate
f0 = Source()
f.interpolate(f0)
File('./output/source.pvd') << f

# Dirichlet boundary conditions (slip conditions)
bc0 = DirichletBC(mixed.sub(1).sub(0), Constant(0.0), LeftRight)
bc1 = DirichletBC(mixed.sub(1).sub(1), Constant(0.0), UpperLower)
bc2 = DirichletBC(mixed.sub(3),        Constant(0.0), Corner, "pointwise")
bcs = [bc0, bc1, bc2]

# Create solution function
U = U0

# List initialisation
E_kinetic = []
momentum_x = []
momentum_y = []

# Matrix and RHS vector
A, b = Matrix(), Vector()

# Time loop
step = 0
t = 0.0
while t < num_steps*float(dt):

    step += 1
    t += float(dt)
    print ' Time step:  ',step, '  time:  ',t

    # Assemble system
    reset_sparsity = (step == 1)
    assemble(a, tensor=A, reset_sparsity=reset_sparsity)
    assemble(L, tensor=b, reset_sparsity=reset_sparsity)

    # Dirichlet bcs
    for bc in bcs: bc.apply(A, b)

    # Solve system
    solve(A, U.vector(), b)

    # write VTK field time level n+theta
    outfile_u << U.split(True)[1]
    outfile_p << U.split(True)[3]

    # Update solution
    U0 = U

    # Zero source term and viscosity after first step
    f.vector()[:] = 0.0
    nu.assign(0.0)

    # Switch theta value
    if step > theta_change_step:
        print "Using theta1 = ", theta1
        theta.assign(theta1)

    # Compute kinetic energy and momentum
    if step > theta_change_step:

        # kinetic energy
        E_kinetic.append( assemble(dot(u0, u0)*dx, mesh=mesh) )
        print "Kinetic energy:", E_kinetic[-1]
        if step > theta_change_step + 1:
           print "        change:", E_kinetic[-1] - E_kinetic[-2]

        # momentum
        ex = as_vector([1.0, 0.0])
        ey = as_vector([0.0, 1.0])
        momentum_x.append(assemble(dot(u0, ex)*dx, mesh=mesh))
        momentum_y.append(assemble(dot(u0, ey)*dx, mesh=mesh))
        print "Momentum:", momentum_x[-1], momentum_y[-1]
        if step > theta_change_step + 1:
            print "  change:", momentum_x[-1] - momentum_x[-2], momentum_y[-1] - momentum_y[-2]

# Write kinetic energy data to file
file_Ek = open("kinetic_energy_P2.dat", "w")
pickle.dump(E_kinetic, file_Ek)
file_Ek.close()
