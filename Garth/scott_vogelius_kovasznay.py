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
p_exact = Expression('-0.5*(1-exp(2*A*x[0]))', A=A, degree=8)


# Geometry parameters
x_min, x_max = -0.5, 1.0
y_min, y_max = -0.5, 1.5


def corners(x, on_boundary):
    return any(near(x[0], X) and near(x[1], Y)
               for X in [x_min, x_max] for Y in [y_min, y_max])

# Pickard, Scott-Vogelius iteration parameters
r = 1E3    # default penalty parameter
tol_sv = 1.e-8
tol_pickard = 1.e-4
iter_max = 100

# Template for files with results
prefix = 'data_scott_vogelius_kovasznay_%d.txt'

# Loop over polynomial degrees
for k in [4, 5][:1]:
    hs = []

    # Loop over meshes
    for n in [16, 32, 64, 128][2:3]:
        M = n/k
        N = 3*M/2
        mesh = RectangleMesh(x_min, y_min, x_max, y_max, M, N)
        h = mesh.hmin()

        # Solve only for velocity. Pressure is obtained in postprocessing
        V = VectorFunctionSpace(mesh, 'CG', k)
        u = TrialFunction(V)
        v = TestFunction(V)

        bc_u = DirichletBC(V, u_exact, DomainBoundary())

        # Penalty parameter
        rho = Constant(r)

        # Vector for divergence term
        w = Function(V)

        # Current and previous solutions
        uh = Function(V)
        u0 = Function(V)

        Re = Constant(Re)

        a = inner(dot(grad(u), u0), v)*dx + 1./Re*inner(grad(u), grad(v))*dx\
            - rho*inner(div(u), div(v))*dx

        L = inner(f, v)*dx + inner(div(w), div(v))*dx

        solver = LUSolver('mumps')

        # S-V and Pickad loops
        iter = 0
        converged = False
        while not converged and iter < iter_max:

            iter += 1
            print  iter

            # Assign previous
            u0.assign(uh)

            # Solve variational problem
            A, b = assemble_system(a, L, bc_u)
            solve(A, uh.vector(), b)

            # Update w
            w.vector().axpy(float(rho), uh.vector())

            # Stopping criteria
            e_pickard, e_sv = None, None
            if iter < 2:
                converged = False
            else:
                e_sv = (uh.vector() - u0.vector()).norm('l2')

                e = norm(uh, 'L2')
                e0 = norm(u0, 'L2')
                e_pickard = abs(e - e0)/(e + e0)

                print e_sv, e_pickard

                converged = e_sv < tol_sv and e_pickard < tol_pickard

        # Compute the pressure
        Q = FunctionSpace(mesh, 'DG', k-1)
        bc_p = DirichletBC(Q, p_exact, corners, 'pointwise')
        ph = project(div(w)/rho, Q, bc_p)

        print sqrt(abs(assemble(div(uh)**2*dx)))

        plot(uh-u_exact)
        plot(ph)
        plot(p_exact, mesh=mesh)
        interactive()

