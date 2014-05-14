'''Solving Stokes flow with Scott-Vogelius penalty.'''

# TODO test: Can k be smaller than 4. Is there a rule for alpha

from dolfin import *

# FFC options as in the original paper
parameters['form_compiler']['cpp_optimize']   = True
parameters['form_compiler']['optimize']       = True
parameters['form_compiler']['representation'] = 'quadrature'

f = Expression(('pi*pi*sin(pi*x[1])-2*pi*cos(2*pi*x[0])',
                'pi*pi*cos(pi*x[0])'))

# Exact solution
u_exact = Expression(('sin(pi*x[1])','cos(pi*x[0])'), degree=8)
p_exact = Expression('sin(2*pi*x[0])', degree=8)
Re = 1.

# Geometry parameters
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0

def corners(x, on_boundary):
    return any(near(x[0], X) and near(x[1], Y)\
            for X in [x_min, x_max] for Y in [y_min, y_max])

# Scott-Vogelius iteration parameters
r = 1.e3 # default value
tol = 1.e-8
iter_max = 4

# Template for files with results
prefix = 'data_scott_vogelius_stokes_%d.txt'

# Loop over polynomial degrees
for k in [1, 2, 3, 4, 5]:
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
        V = VectorFunctionSpace(mesh, 'CG', k)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Penalty parameter
        r = Constant(r)
        # Vector for divergence term
        w = Function(V)

        # Current and previous solutions
        Uh = Function(V)
        U0 = Function(V)

        Re = Constant(Re)
        a = 1./Re*inner(grad(u), grad(v))*dx + r*inner(div(u), div(v))*dx
        L = inner(f, v)*dx - inner(div(w), div(v))*dx

        bc_u = DirichletBC(V, u_exact, DomainBoundary())

        # S-V loop
        solver = LUSolver('mumps')
        converged = False
        while not converged:
            print 'Using penalty parameter', float(r)
            iter = 0
            while iter < iter_max:
                A, b = assemble_system(a, L, bc_u)

                iter += 1

                print iter

                # Previous solution
                U0.assign(Uh)

                # Solve variational problem
                solver.solve(A, Uh.vector(), b)

                # Updata w
                w.vector().axpy(float(r), Uh.vector())

                # Stopping criteria
                e = None
                if iter < 2:
                    converged = False
                else:
                    e = (Uh.vector() - U0.vector()).norm('l2')
                    converged = e < tol
                print '\t', e
                if converged : break

            # Run againg with new penalty parameter
            r.assign(float(r)*10)
            print 'Increased penalty parameter %g\n' % float(r)

        # Store the final penalty and iteration count
        penalties.append((float(r)/10, iter))

        # Compute L2 velocity error
        u_diff = Uh - u_exact
        u_error = sqrt(abs(assemble(dot(u_diff, u_diff)*dx, mesh=mesh)))

        # Compute the pressure and its L2 error
        Q = FunctionSpace(mesh, 'DG', k-1)
        bc_p = DirichletBC(Q, p_exact, corners, 'pointwise')
        ph = project(div(w), Q, bc_p)

        p_diff = ph- p_exact
        p_error = sqrt(abs(assemble(p_diff*p_diff*dx, mesh=mesh)))

        # Compute divergence error
        div_error = sqrt(abs(assemble(div(Uh)*div(Uh)*dx, mesh=mesh)))

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

    penalty_file = 'penalties_scott_vogelius_stokes_%d.txt' % k
    with open(penalty_file, 'w') as file:
        for h, (a, n) in zip(hs, penalties):
            file.write('%e %e(%d)\n' % (h, a, n))
