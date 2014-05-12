'''Solving Kovasznay flow with Taylor-Hood P_{k+1} - P_{k} element.'''

# TODO test

import matplotlib.pyplot as plt
from dolfin import *

# FFC options as in the original paper
parameters['form_compiler']['cpp_optimize']   = True
parameters['form_compiler']['optimize']       = True
parameters['form_compiler']['representation'] = 'quadrature'

# Exact solution depends on Re
Re = 40.

A = Re/2 - pow(pow(Re/2, 2) + pow(2*pi, 2), 0.5)
u_exact = Expression(('1-exp(A*x[0])*cos(2*pi*x[1])',
                      'A*exp(A*x[0])*sin(2*pi*x[1])/(2*pi)'), A=A, degree=8)
p_exact = Expression('-0.5*(1-exp(2*A*x[0]))', A=A, degree=8)

# Geometry parameters
x_min, x_max = -0.5, 1.0
y_min, y_max = -0.5, 1.5

def corners(x, on_boundary):
    return any(near(x[0], X) and near(x[1], Y)\
            for X in [x_min, x_max] for Y in [y_min, y_max])

# Picard fixed point iteration parameters
tol = 1.e-4
iter_max = 50

# Template for files with results
prefix = 'data_th_kowasznay_%d.txt'

# Loop over polynomial degrees
for k in [1, 2, 3, 4]:
    error_u = []
    error_p = []
    error_div = []
    hs = []
    
    # Loop over meshes
    for n in [16, 32, 64, 128]:
        M = n/k
        N = 3*M/2
        mesh = RectangleMesh(x_min, y_min, x_max, y_max, M, N)
        h = mesh.hmin()

        V = VectorFunctionSpace(mesh, 'CG', k+1)
        Q = FunctionSpace(mesh, 'CG', k)
        M = MixedFunctionSpace([V, Q])

        u, p = TrialFunctions(M)
        v, q = TestFunctions(M)

        # Solution at current iteration
        Uh = Function(M)

        # Solution at previous iteration
        U0 = Function(M)
        u0, p0 = split(U0)

        Re = Constant(Re)
        f = Constant((0., 0.))

        F = inner(dot(grad(u), u0), v)*dx + 1./Re*inner(grad(u), grad(v))*dx +\
                inner(div(v), p)*dx + inner(div(u), q)*dx + inner(f, v)*dx
        a, L = system(F)

        bc_u = DirichletBC(M.sub(0), u_exact, DomainBoundary())
        bc_p = DirichletBC(M.sub(1), p_exact, corners, 'pointwise')
        bcs = [bc_u, bc_p]

        # Fixed point iteration
        converged = False   
        iter = 0
        err_u = []
        err_p = []
        err_div = []

        solver = LUSolver('mumps')
        while converged == False and iter < iter_max:
            A, b = assemble_system(a, L, bcs)

            iter += 1

            print iter

            # Previous solution
            U0.assign(Uh)

            # Solve variational problem
            solver.solve(A, Uh.vector(), b)

            # Relaxation
            Uh.vector()[:]=(U0.vector()[:] + Uh.vector()[:])/2.

            # Get solution sub-functions
            uh = Uh.split()[0]
            ph = Uh.split()[1]

            # Compute L2 velocity error
            u_diff = uh - u_exact
            u_error = sqrt(abs(assemble(dot(u_diff, u_diff)*dx, mesh=mesh)))
            err_u.append(u_error)

            # Compute L2 pressure error
            p_diff = ph- p_exact
            p_error = sqrt(abs(assemble(p_diff*p_diff*dx, mesh=mesh)))
            err_p.append(p_error)

            # Compute divergence error
            err_div.append(sqrt(abs(assemble(div(uh)*div(uh)*dx, mesh=mesh))))

            # Stopping criteria
            e = None
            if iter < 2:
                converged = False
            else:
                e1 = err_u[-1]
                e2 = err_u[-2]
                e = abs(e1 - e2)/(e1 + e2)
                converged = e < tol

            print '\terror_u     = ', err_u[-1]
            print '\terror_p     = ', err_p[-1]
            print '\terror_div_u = ', err_div[-1]
            print '\te           = ', e

        # Store the norms for h
        error_u.append(err_u[-1])
        error_p.append(err_p[-1])
        error_div.append(err_div[-1])
        hs.append(h)
    
    # Store results to files
    data_file = prefix % k
    with open(data_file, 'w') as file:
        for row in zip(hs, error_u, error_p, error_div):
            file.write('%e %e %e %e\n' % row)
