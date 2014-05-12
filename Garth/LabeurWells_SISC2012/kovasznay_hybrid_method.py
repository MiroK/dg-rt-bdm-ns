""" Incompressible Navier-Stokes equations
for Kovasznay flow convergence test;
using hybrid method """

__author__    = "Robert Jan Labeur and Garth N. Wells"
__date__      = "11-11-2011"
__copyright__ = "Copyright (C) 2010-2011 Robert Jan Labeur and Garth N. Wells"
__license__   = "GNU LGPL Version 3"

from dolfin import *

# Code generation options
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["representation"] = "quadrature"

# plot files
outfile_u = File('velo.pvd')
outfile_p = File('pres.pvd')

# Error file
error_file = open("errors.out","w")
print >> error_file, 5*'%25s' % ('polynomial order', 'cell width (he)',\
     'velocity error','pressure error','divergence error')

# Reynolds number
Re = 40.0

# Viscosity
nu = 1.0/Re

# advection scheme (0) advective, (1) conservative
chi = .5 

# Exact solution
La = Re/2 - pow(pow(Re/2, 2) + pow(2*pi, 2), 0.5)
P7 = VectorElement("Lagrange", "triangle", 8)
P2 = FiniteElement("Lagrange", "triangle", 8)
u_exact = Expression(("1-exp(A*x[0])*cos(2*pi*x[1])",\
                      "A*exp(A*x[0])*sin(2*pi*x[1])/(2*pi)"), \
                       A = La, element = P7)
p_exact = Expression("0.5*(1-exp(2*A*x[0]))", A = La, element = P2)

# Mesh sizes
M = [16, 32, 64, 128]

# Loop over polynomial order
for k in [1, 2, 3, 4, 5]:

    # initialize error lists
    error_u = []
    error_p = []
    error_div = []

    # loop over mesh size
    for N in M:

        NN = N/k
        print "k=", k, " N=", N, " NN=", NN

        # Create mesh
        mesh = Rectangle(-.5, -.5, 1., 1.5, NN, 3*NN/2)

	# Define boundary
	def Gamma(x, on_boundary): return on_boundary

        # Create function spaces
        V    = VectorFunctionSpace(mesh, 'DG', k)
        Vbar = VectorFunctionSpace(mesh, 'CG', k, restriction='facet')
        Q    = FunctionSpace(mesh, 'DG', k)
        Qbar = FunctionSpace(mesh, 'CG', k, restriction='facet')

        # Create mixed function space
        mixed = MixedFunctionSpace([V, Vbar, Q, Qbar])

        # Create trial and test functions
        v, vbar, q, qbar = TestFunctions(mixed)
        u, ubar, p, pbar = TrialFunctions(mixed)

        # auxiliary function
        Uh = Function(mixed)

        # Initial condition
        U0 = Function(mixed)
        u0, ubar0, p0, pbar0 = split(U0)

        # Source term
        f = Constant( (0.0, 0.0) )

        # Penalty parameters
        alpha = Constant(6*k*k)
        beta = 1.0e-4

        # Mesh related functions
        n  = FacetNormal(mesh)
        he = CellSize(mesh)

        # Dirichlet boundary conditions using exact solution
        bc0 = DirichletBC(mixed.sub(1), u_exact, Gamma)
        bc1 = DirichletBC(mixed.sub(3), p_exact, Gamma)
        bcs = [bc0, bc1]

        # Interface mass flux
        uhat_n       = dot(u, n) - (beta*he/(1+nu))*(pbar - p)
        uhat0_n      = dot(u0, n) - (beta*he/(1+nu))*(pbar0 - p0)
        lambda_uhat0 = (uhat0_n - abs(uhat0_n))/2.0

        # Exterior boundary mass flux
        ubar0_n      = dot(ubar0, n)
        lambda_ubar0 = (ubar0_n - abs(ubar0_n))/2.

        # Momentum fluxes
        sigma_d   = p*Identity(V.cell().d) - 2*nu*sym(grad(u))
        sigma_n_d = pbar*n - 2*nu*sym(grad(u))*n - (2*nu*alpha/he)*sym(outer(ubar - u,n))*n

        # Short-cut function for evaluating sum_{K} \int_{K} (integrand) ds
        def facet_integral(integrand):
            return integrand('-')*dS + integrand('+')*dS + integrand*ds

        # Functionals (solving F = 0)
        F_m_local_a = - chi*inner(outer(u, u0), grad(v))*dx \
                    + (1.0 - chi)*dot(grad(u)*u0, v)*dx \
                    + chi*facet_integral( uhat0_n*dot(u, v) ) \
                    + facet_integral( lambda_uhat0*dot(ubar-u, v) )
        F_m_local_d  =  -inner(sigma_d, grad(v))*dx \
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

        # fixed point iteration
        tol = 1.e-4
        converged = False   
        iter = 0
        err_u = []
        err_p = []
        err_div = []

        while converged == False and iter < 50:
            iter+=1

	    print iter

            # previous solution
            U0.assign(Uh)

	    # Bilinear and linear forms
            a = F
            L = dot(f,v)*dx

            # Solve variational problem
            solve(a == L, Uh, bcs)

            # relaxation
            Uh.vector()[:]=(U0.vector()[:]+Uh.vector()[:])/2

            # Get solution sub-functions
            uh = Uh.split()[0]
            ph = Uh.split()[2]

            # Compute L2 velocity error
            u_error = uh - u_exact
            err_u.append(sqrt(abs(assemble(dot(u_error, u_error)*dx, mesh=mesh))))

            # Compute L2 pressure error
            p_error = ph - p_exact
            err_p.append(sqrt(abs(assemble(p_error*p_error*dx, mesh=mesh))))

            # Compute divergence error
            err_div.append(sqrt(abs(assemble(div(uh)*div(uh)*dx, mesh=mesh))))

            print " error_u     = ", err_u[-1]
       	    print " error_p     = ", err_p[-1]
            print " error_div_u = ", err_div[-1]

            # stopping criteria
            if iter < 2:
                converged = False
            else:
                e1 = err_u[-1]
                e2 = err_u[-2]
                converged = abs(e1-e2)/(e1+e2) < tol
	
        # store L2 errors
        error_u.append(err_u[-1])
        error_p.append(err_p[-1])
        error_div.append(err_div[-1])
            
        # plot fields
        outfile_u << Uh.split(True)[1]
        outfile_p << Uh.split(True)[3]

    # Write error results to file
    for m, eu, ep, ed in zip(M, error_u, error_p, error_div):
       	print >> error_file, 5*'%25.20f' % (k, 1/(m/k), eu, ep, ed)

