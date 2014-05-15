from dolfin import *
import time

# TODO
# Add biharmonic
# Test effect of stab. parameters of DG problem


def eigenproblem_solver(a_form, m_form, V, bcs=None, n=None):
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
    print 'Done in %gs.' % t.timing

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
    print 'Done in %gs.' % t.timing

    i = 0
    n_returned = min(n, m_max, m_min)
    while i < n_returned:
        yield eigensolver_min.get_eigenvalue(i),\
            eigensolver_max.get_eigenvalue(i)
        i += 1


def eigenproblem_lambda_min_max(a_form, m_form, V, bcs=None):
    'Compute largest and smallest eigenvalue of gener. eigenvalue problem.'

    # Get eigenvalues. Take n_eigs to see some trends maybe.
    n_eigs = 3
    eigs = eigenproblem_solver(a_form, m_form, V, bcs=bcs, n=n_eigs)
    lambda_min, lambda_max = [], []
    for x, y in eigs:
        lambda_min.append(x)
        lambda_max.append(y)

    # Postproces
    ZERO = 1e-10
    is_zero = lambda z: abs(z) < ZERO
    is_real = lambda z: is_zero(z[1])
    get_real = lambda z: z[0]

    # Warn if there are nonzero imag parts
    if len(map(get_real, filter(is_real, lambda_min))) != n_eigs or\
            len(map(get_real, filter(is_real, lambda_max))) != n_eigs:
        print 'Complex spectrum!'

    # Keep only real part of spectrum
    lambda_min = map(get_real, lambda_min)
    lambda_max = map(get_real, lambda_max)

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
    h = mesh.hmin()
    numeric = eigenproblem_lambda_min_max(a_form, m_form, V, bcs=[bc])
    exact = pi**2, (pi/h)**2
    # No bcs -- constant vector in the nullspace, i.e. there is lambda = 0
    # With bcs -- there is some scaling difference but okay for now

    print 'Numeric: %.2f %.2f' % numeric
    print 'Exact:   %.2f %.2f' % exact


class Timeit(object):
    'Simple timer context manager'
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.timing = time.time() - self.start
        return False

if __name__ == '__main__':
    poisson_eigenvalues()
