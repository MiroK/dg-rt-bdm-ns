'''Solving Stokes flow with Scott-Vogelius penalty.'''

# TODO test: Can k be smaller than 4. Is there a rule for alpha

from dolfin import *

# FFC options as in the original paper
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['representation'] = 'quadrature'

f = Constant((0., 0.))

# Exact solution
Re = 40.

A = Re/2 - pow(pow(Re/2, 2) + pow(2*pi, 2), 0.5)
u_exact = Expression(('1-exp(A*x[0])*cos(2*pi*x[1])',
                      'A*exp(A*x[0])*sin(2*pi*x[1])/(2*pi)'), A=A, degree=8)
p_exact = Expression('0.5*(1-exp(2*A*x[0]))', A=A, degree=8)


# Geometry parameters
x_min, x_max = -0.5, 1.0
y_min, y_max = -0.5, 1.5


def corners(x, on_boundary):
    return any(near(x[0], X) and near(x[1], Y)
               for X in [x_min, x_max] for Y in [y_min, y_max])

# Scott-Vogelius iteration parameters
r = 1E3       # default penalty parameter
r_max = 1e12  # maximum value of penalty parameter
tol = 1.e-8
iter_max = 100

# Template for files with results
prefix = 'data_scott_vogelius_kovasznay_%d.txt'

# Loop over polynomial degrees
for k in [1]:
    error_u = []
    error_p = []
    error_div = []
    hs = []
    penalties = []

    # Loop over meshes
    for n in [16, 32, 64, 128]:
        M = n/k
        N = 3*M/2
        mesh = RectangleMesh(x_min, y_min, x_max, y_max, M, N)
        h = mesh.hmin()

        # Solve only for velocity. Pressure is obtained in postprocessing
        V = VectorFunctionSpace(mesh, 'CR', k)
        u = Function(V)
        v = TestFunction(V)

        # Boundary conditions
        bc_u = DirichletBC(V, u_exact, DomainBoundary())

        # Penalty parameter
        rho = Constant(r)

        # Divergence sum
        w = Function(V)

        F = inner(div(outer(u, u)), v)*dx + 1./Re*inner(grad(u), grad(v))*dx \
            - rho*inner(div(u), div(v))*dx - inner(div(w), div(v))*dx -\
            inner(f, v)*dx

        # Current and previous solutions in S-V loop
        u0 = Function(V)

        # Scott-Vogelius loop
        converged = False
        r_ = r  # remember the degault penalty
        while not converged and r_ < r_max:
            print 'Using penalty parameter %.2e\n' % r_
            iter = 0

            while iter < iter_max:

                iter += 1
                print iter

                # Assign previous
                u0.assign(u)

                # Solve variational problem
                try:
                    solve(F == 0, u, bc_u,
                          solver_parameters=
                          {'newton_solver': {'relative_tolerance': 1E-6,
					     'absolute_tolerance': 1E-6,
                                             'linear_solver': 'mumps'}})
                except RuntimeError:
                    u.vector().zero()
                    u0.vector().zero()
                    break

                # Update w
                w.vector().axpy(float(rho), u.vector())

                # Stopping criteria
                e_sv = (u.vector() - u0.vector()).norm('l2')
                converged = e_sv < tol
                print '\t', e_sv

                if converged:
                    break

            # Run againg with new penalty parameter
            if not converged:
                r_ *= 10
                rho.assign(r_)
                w.vector().zero()
                print 'Increased penalty parameter %.2e\n' % r_

        # Store the final penalty and iteration count
        penalties.append((r_, iter))

        # Compute the pressure
        Q = FunctionSpace(mesh, 'DG', 0)

	# With velocity k=4, 5 in CG, the pressure in CG/DG k-1 is not correct!
	# With celocity k=2, 3 in CG, pressure in CG k-1 seems most reasonable,
	# that is there are refinements where the pressure convgerges.
        # Also, CR1-DG0 pair does not deliver. 
	# Overall, the penalty approch seems more suitable for stokes then
	# navier-stokes
        # RT 1, 2, 3 and DG k-1 is wrong also
        bc_p = DirichletBC(Q, p_exact, corners, 'pointwise')
        p = project(div(w), Q, bc_p)

        # Compute L2 velocity error
        u_diff = u - u_exact
        u_error = sqrt(abs(assemble(dot(u_diff, u_diff)*dx, mesh=mesh)))

        # Compute L2 pressure error
        p_diff = p - p_exact
        p_error = sqrt(abs(assemble(p_diff*p_diff*dx, mesh=mesh)))

        # Compute divergence error
        div_error = sqrt(abs(assemble(div(u)*div(u)*dx, mesh=mesh)))

        # Store the norms for h
        error_u.append(u_error)
        error_p.append(p_error)
        error_div.append(div_error)
        hs.append(h)

    # Store results to files
    data_file = prefix % k
    with open(data_file, 'w') as file:
        for row in zip(hs, error_u, error_p, error_div):
            file.write('%e %e %e %e\n' % row)

    penalty_file = 'penalties_scott_vogelius_kovasznay_%d.txt' % k
    with open(penalty_file, 'w') as file:
        for h, (a, n) in zip(hs, penalties):
            file.write('%e %e %d\n' % (h, a, n))
