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
r = 1E3    # default penalty parameter
tol = 1.e-8
iter_max = 100

# Template for files with results
prefix = 'data_scott_vogelius_kovasznay_%d.txt'

# Loop over polynomial degrees
for k in [4, 5]:
    hs = []

    # Loop over meshes
    for n in [16, 32, 64, 128]:
        M = n/k
        N = 3*M/2
        mesh = RectangleMesh(x_min, y_min, x_max, y_max, M, N)
        h = mesh.hmin()

        # Solve only for velocity. Pressure is obtained in postprocessing
        V = VectorFunctionSpace(mesh, 'CG', k)
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

        # S-V and Pickad loops
        iter = 0
        converged = False
        while not converged and iter < iter_max:

            iter += 1
            print  iter

            # Assign previous
            u0.assign(u)

            # Solve variational problem
            solve(F == 0, u, bc_u)

            # Update w
            w.vector().axpy(float(rho), u.vector())

            # Stopping criteria
            e_sv = None
            if iter < 2:
                converged = False
            else:
                e_sv = (u.vector() - u0.vector()).norm('l2')

                print e_sv

                converged = e_sv < tol

        # Compute the pressure
        Q = FunctionSpace(mesh, 'DG', k-1)
        bc_p = DirichletBC(Q, p_exact, corners, 'pointwise')
        ph = project(div(w), Q, bc_p)

        print sqrt(abs(assemble(div(u)**2*dx)))

        plot(u-u_exact)
        plot(ph)
        plot(p_exact, mesh=mesh)
        interactive()

