from dolfin import *
import time

# TODO
# Test effect of stab. parameters of DG problem


def eigenproblem_solver(a_form, m_form, V, bcs=None, n=None, symmetric=True):
    '''Get (n) largest and smallest eigenvaues of generalized eigenvalue
    problem'''

    # Build the matrix problem A*U = lambda*M*U
    # Generalized eigenvalue problem is built as
    # a_form(u, v) = lambda*m_form(u, v) where u, v are
    # Trial/Test-Function of V
    u = TrialFunction(V)
    v = TestFunction(V)

    A = PETScMatrix()
    M = PETScMatrix()
    b = PETScVector()

    if symmetric:
        # Fake linear form for assemble_system
        L = lambda v: inner(Constant(0, V.cell()), v)*dx

        assemble_system(a_form(u, v), L(v), bcs, A_tensor=A, b_tensor=b)
        assemble_system(m_form(u, v), L(v), bcs, A_tensor=M, b_tensor=b)
    else:
        assemble(a_form(u, v), tensor=A)
        assemble(m_form(u, v), tensor=M)

        if bcs is not None:
            assert isinstance(bcs, list)
            for bc in bcs:
                bc.apply(A)
                bc.apply(M)

    # Get number of eigs to compute
    N = A.size(0)
    if n is None:
        n = N/100 + 1  # Go for ~1% of spectrum
    else:
        # If something reasonable was given use it, otherwise compute all eigs
        n = n if n < N else N

    # Get smallest eigs, in particular we are worried about zeros
    eigensolver_min = SLEPcEigenSolver(A, M)
    params = eigensolver_min.parameters
    params['solver'] = 'arpack'
    params['spectrum'] = 'target magnitude'
    params['spectral_transform'] = 'shift-and-invert'
    params['spectral_shift'] = 1e-6

    print 'Computing %d smallest eigenvalues:' % n
    with Timeit() as t:
        eigensolver_min.solve(n)
        m_min = eigensolver_min.get_number_converged()
    print 'Done in %gs.\n' % t.timing

    # Get largest eigs
    eigensolver_max = SLEPcEigenSolver(A, M)
    params = eigensolver_max.parameters
    params['solver'] = 'arpack'
    params['spectrum'] = 'largest magnitude'
    params['spectral_transform'] = 'default'
    params['spectral_shift'] = 0.0

    print 'Computing %d largest eigenvalues:' % n
    with Timeit() as t:
        eigensolver_max.solve(n)
        m_max = eigensolver_max.get_number_converged()
    print 'Done in %gs.\n' % t.timing

    i = 0
    n_returned = min(n, m_max, m_min)
    while i < n_returned:
        yield eigensolver_min.get_eigenvalue(i),\
            eigensolver_max.get_eigenvalue(i)
        i += 1


def eigenproblem_lambda_min_max(a_form, m_form, V, bcs=None):
    'Compute largest and smallest eigenvalue of gener. eigenvalue problem.'

    # Get eigenvalues. Take n_eigs to see some trends maybe.
    n_eigs = 8
    eigs = eigenproblem_solver(a_form, m_form, V, bcs=bcs, n=n_eigs)
    lambda_min, lambda_max = [], []
    for x, y in eigs:
        lambda_min.append(x)
        lambda_max.append(y)

    # Postproces
    ZERO = 1e-3*V.mesh().hmin()
    is_zero = lambda z: abs(z) < ZERO
    is_real = lambda z: is_zero(z[1])
    get_real = lambda z: z[0]

    # Warn if there are nonzero imag parts
    if len(map(get_real, filter(is_real, lambda_min))) != n_eigs or\
            len(map(get_real, filter(is_real, lambda_max))) != n_eigs:
        print 'Complex spectrum!'

    # Keep only real part of spectrum
    lambda_min = sorted(map(get_real, lambda_min), key=abs)
    lambda_max = sorted(map(get_real, lambda_max), key=abs, reverse=True)

    print 'lambda min', lambda_min
    print 'lambda max', lambda_max

    if max(abs(lambda_min[0]), abs(lambda_max[0])) < ZERO:
        print 'Zero spectrum'
        return 0., 0.
    else:
        nnz = n_eigs - len(filter(is_zero, lambda_min))
        # Not singular case
        if nnz == n_eigs:
            print 'No zero eigenvalues'
            return lambda_min[0], lambda_max[0]

        # If there is single 0 eigenvalue, this is likely due to missing bcs
        elif nnz == n_eigs - 1:
            print 'Single 0 eigenvalue. Missing Dirichlet bcs?'
            return lambda_min[1], lambda_max[0]

        else:
            print 'At least %d vectors in the nullspace' % (n_eigs-nnz)
            return 0., lambda_max[0]


def poisson_eigenvalues(mesh_size=1000):
    'Eigenvalues of -laplace(u) = lambda*u in [0, 1]'

    a_form = lambda u, v: inner(grad(u), grad(v))*dx
    m_form = lambda u, v: inner(u, v)*dx

    mesh = UnitIntervalMesh(mesh_size)
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    # Exact eigenvalues of continous well-posed problem are (k*pi)**2
    # k = 1, k = (2*pi/2*h) = pi/h
    h = mesh.hmin()
    numeric = eigenproblem_lambda_min_max(a_form, m_form, V, bcs=[bc])
    exact = pi**2, (pi*pi/h)**2
    # No bcs -- constant vector in the nullspace, i.e. there is lambda = 0
    # Symmetry in applying bcs does not seem to change eigenvalues at all
    # Interesting that the spectrum is (ones, smallest analytic, ...)
    # and there is about factor 8 difference ... maybe exact continuous vs.
    # exact discrete
    # Ones correspond to eigenvectors which can't be resolved by mesh, i.e.
    # they have too small wavelength (less then 2*h)

    print 'Numeric: %.6g %.2g' % numeric
    print 'Exact:   %.6g %.2g' % exact


def biharmonic_eigenvalues(alpha_value, mesh_size=32):
    'Eigenvalues of del**4 = lambda*u in [0, 1]**2'

    mesh = UnitSquareMesh(mesh_size, mesh_size)
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(mesh)

    # Penalty parameter
    alpha = Constant(alpha_value)

    a_form = lambda u, v: inner(div(grad(u)), div(grad(v)))*dx \
        - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
        - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
        + alpha('+')/h_avg*inner(jump(grad(u), n), jump(grad(v), n))*dS

    m_form = lambda u, v: inner(u, v)*dx

    V = FunctionSpace(mesh, 'CG', 2)
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    numeric = eigenproblem_lambda_min_max(a_form, m_form, V, bcs=None)
    # Symmetry in applying bcs does not seem to change eigenvalues at all
    # Interesting that the spectrum is (3*onnes, smallest analytic, ...)
    # its probably for same reasons as poisson
    # No bcs -- problem should be singular - it's apperant in 2d,
    # not so much in 1d

    # See the effect of alpha on the solution
    biharmonic_problem(alpha_value, mesh_size)

    print '%.6g %.2g' % numeric


def biharmonic_problem(alpha_value, mesh_size):
    'Demo ...'
    # Create mesh and define function space
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, "CG", 2)

    class Source(Expression):
        def eval(self, values, x):
            values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

    # Define boundary condition
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define normal component, mesh size and right-hand side
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(mesh)
    f = Expression('4.0*pi*pi*pi*pi*sin(pi*x[0])*sin(pi*x[1])')

    # Penalty parameter
    alpha = Constant(alpha_value)

    # Define bilinear form
    a = inner(div(grad(u)), div(grad(v)))*dx\
        - inner(avg(div(grad(u))), jump(grad(v), n))*dS\
        - inner(jump(grad(u), n), avg(div(grad(v))))*dS\
        + alpha('+')/h_avg*inner(jump(grad(u), n), jump(grad(v), n))*dS

    # Define linear form
    L = f*v*dx

    # Solve variational problem
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution
    plot(u, interactive=True, title='%g' % alpha_value)


class Timeit(object):
    'Simple timer context manager'
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.timing = time.time() - self.start
        return False


def dg_poisson_eigenvalues(alpha_value, mesh_size=32):
    'Eigenvalues of -laplace(u) = lambda*u in [0, 1]'

    mesh = UnitIntervalMesh(mesh_size)

    # Define DG parameters
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    alpha = Constant(alpha_value)
    gamma = Constant(2*alpha_value)

    a_form = lambda u, v: dot(grad(v), grad(u))*dx\
        - dot(avg(grad(v)), jump(u, n))*dS\
        - dot(jump(v, n), avg(grad(u)))*dS\
        + avg(alpha)/avg(h)*dot(jump(v, n), jump(u, n))*dS\
        - dot(grad(v), u*n)*ds\
        - dot(v*n, grad(u))*ds\
        + gamma/h*v*u*ds

    m_form = lambda u, v: (inner(grad(u), grad(v)))*dx\
        + 1.0/(avg(h))*dot(jump(v, n), jump(u, n))*dS \
        + 1.0/h*dot(v*n, u*n)*ds

    V = FunctionSpace(mesh, 'DG', 1)

    numeric = eigenproblem_lambda_min_max(a_form, m_form, V, bcs=None)

    # Witness the effect
    dg_poisson_problem(alpha_value, mesh_size)

    print 'Numeric: %.6g %.2g' % numeric


def dg_poisson_problem(alpha_value, mesh_size):
    'Demo ...'
    # Create mesh and define function space
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, 'DG', 1)

    # Define test and trial functions
    v = TestFunction(V)
    u = TrialFunction(V)

    # Define normal component, mesh size and right-hand side
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    f = Expression('500.0*exp(-(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2)) / 0.02)')

    # Define parameters
    alpha = Constant(alpha_value)
    gamma = Constant(2*alpha_value)

    # Define variational problem
    a = dot(grad(v), grad(u))*dx\
        - dot(avg(grad(v)), jump(u, n))*dS\
        - dot(jump(v, n), avg(grad(u)))*dS\
        + avg(alpha)/h_avg*dot(jump(v, n), jump(u, n))*dS\
        - dot(grad(v), u*n)*ds\
        - dot(v*n, grad(u))*ds\
        + gamma/h*v*u*ds

    L = v*f*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u)

    # Project solution to piecewise linears
    u_proj = project(u)
    # Plot solution
    plot(u_proj, interactive=True, title='%g' % alpha_value)


if __name__ == '__main__':
    # Test
    # poisson_eigenvalues()

    # This has no bcs on purpose!
    # for alpha 1, 0 there are complex eigs and there are negative lmbda max
    # Only for alpha=0 lmbda_min = 1E-16 (until then about 1E-9) and the
    # solution has visible oscillations
    # for alpha in [128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
    #    print 'alpha =', alpha
    #    biharmonic_eigenvalues(alpha_value=alpha)
    #    print

    # Only alpha=0,1 have zeros in spectrum
    # alpha=0 ---> 1E-11, larger oscllation near bdr
    # alpah=1 ---> 1E-17, very small oscillation near bdr
    # Only alpha=0 has negative lambda_min
    for alpha in [128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        print 'alpha =', alpha
        dg_poisson_eigenvalues(alpha_value=alpha)
        print
