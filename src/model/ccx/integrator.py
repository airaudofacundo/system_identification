import numpy as np
from scipy.special import roots_legendre, eval_legendre

class integrator:

    gauss_order = 0
    integ_terms = 0
    ndim = 0

    pdf_type = 0

    def __init__(self, gauss_order, integ_type, ndim=0):
        self.gauss_order = gauss_order
        self.value_gpoints(integ_type, ndim)

    def value_gpoints(self, integ_type, ndim=0):
        if integ_type.strip().lower() in ['line', '1d']:
            self.getG1D()
        elif integ_type.strip().lower() in ['triangle', 'triang']:
            self.getGTriangle()
        elif integ_type.strip().lower() in ['quadrilateral', 'quad']:
            self.getGSquare()
        elif integ_type.strip().lower() in ['tetrahedron', 'tetra']:
            self.getGTetrahedron()
        elif integ_type.strip().lower() in ['hexahedron', 'hexa']:
            self.getGHexahedron()
        elif integ_type.strip().lower() in ['hypercube', 'hyper']:
            self.getGHypercube(ndim)
        else:
            raise ValueError('Unrecognized integ_type')

    def getG1D(self):
        self.gpoint, self.weight = roots_legendre(self.gauss_order)
        self.integ_terms = self.gauss_order

    def getGTriangle(self):
        gauss_order = self.gauss_order
        if gauss_order == 1:
            integ_terms = 1
            self.weight = np.array([1.0 / 2.0], dtype=float)
            self.gpoint = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        elif gauss_order == 2:
            integ_terms = 3
            self.weight = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)
            self.gpoint = np.array([[1.0 / 2.0, 0.0], [1.0 / 2.0, 1.0 / 2.0], [0.0, 1.0 / 2.0]], dtype=float)
        elif gauss_order == 3:
            integ_terms = 4
            self.weight = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0], dtype=float)
            self.gpoint = np.array([[1.0 / 3.0, 1.0 / 3.0], [1.0 / 5.0, 1.0 / 5.0],
                                    [3.0 / 5.0, 1.0 / 5.0], [1.0 / 5.0, 3.0 / 5.0]], dtype=float)
        else:
            raise ValueError('** Input Gauss Order not supported for triangular elements! **')
        self.integ_terms = integ_terms

    def getGSquare(self):
        gpoint, weight = roots_legendre(self.gauss_order)
        self.integ_terms = self.gauss_order**2
        self.weight = np.zeros(self.integ_terms, dtype=np.float64)
        self.gpoint = np.zeros((self.integ_terms, 2), dtype=np.float64)
        counter = -1
        for i in range(self.gauss_order):
            for j in range(self.gauss_order):
                counter += 1
                self.weight[counter] = weight[i]*weight[j]
                self.gpoint[counter][0] = gpoint[i]
                self.gpoint[counter][1] = gpoint[j]

    def getGTetrahedron(self):
        gauss_order = self.gauss_order
        if gauss_order == 1:
            integ_terms = 1
            self.weight = np.array([1.0], dtype=float)
            self.gpoint = np.array([[0.25, 0.25, 0.25]], dtype=float)
        elif gauss_order == 2:
            integ_terms = 5
            self.weight = np.array([-0.8, 0.45, 0.45, 0.45, 0.45], dtype=float)
            self.gpoint = np.array([[0.25, 0.25, 0.25],
                                    [0.166666666666667, 0.166666666666667, 0.166666666666667],
                                    [0.5, 0.166666666666667, 0.166666666666667],
                                    [0.166666666666667, 0.5, 0.166666666666667],
                                    [0.166666666666667, 0.166666666666667, 0.5]], dtype=float)
        elif gauss_order == 3:
            integ_terms = 11
            self.weight = np.array([0.013155555555555, 0.007622222222222, 0.007622222222222,
                                    0.007622222222222, 0.007622222222222, 0.024888888888888,
                                    0.024888888888888, 0.024888888888888, 0.024888888888888,
                                    0.024888888888888, 0.024888888888888], dtype=float)
            self.gpoint = np.array([[0.25, 0.25, 0.25],
                                    [0.0714285714285714, 0.0714285714285714, 0.785714285714286],
                                    [0.0714285714285714, 0.0714285714285714, 0.0714285714285714],
                                    [0.785714285714286, 0.0714285714285714, 0.0714285714285714],
                                    [0.0714285714285714, 0.785714285714286, 0.0714285714285714],
                                    [0.399403576166799, 0.100596423833201, 0.100596423833201],
                                    [0.399403576166799, 0.399403576166799, 0.100596423833201],
                                    [0.100596423833201, 0.399403576166799, 0.399403576166799],
                                    [0.100596423833201, 0.100596423833201, 0.399403576166799],
                                    [0.100596423833201, 0.399403576166799, 0.100596423833201],
                                    [0.399403576166799, 0.100596423833201, 0.399403576166799]], dtype=float)
        else:
            raise ValueError('** Input Gauss Order not supported for tetrahedral elements! **')
        self.integ_terms = integ_terms

    def getGHexahedron(self):
        gpoint, weight = roots_legendre(self.gauss_order)
        self.integ_terms = self.gauss_order**3
        self.weight = np.zeros(self.integ_terms, dtype=np.float64)
        self.gpoint = np.zeros((self.integ_terms, 3), dtype=np.float64)
        counter = -1
        for i in range(self.gauss_order):
            for j in range(self.gauss_order):
                for k in range(self.gauss_order):
                    counter += 1
                    self.weight[counter] = weight[i]*weight[j]*weight[k]
                    self.gpoint[counter][0] = gpoint[i]
                    self.gpoint[counter][1] = gpoint[j]
                    self.gpoint[counter][2] = gpoint[k]

    def getGHypercube(self, ndim):
        gpoint, weight = roots_legendre(self.gauss_order)
        self.integ_terms = self.gauss_order**ndim
        self.ndim = ndim
        self.weight = np.ones(self.integ_terms, dtype=np.float64)
        self.gpoint = np.zeros((self.integ_terms, ndim), dtype=np.float64)
        counter = -1
        idx = np.zeros(ndim, dtype=np.int64)
        idx[ndim-1] = -1
        for igauss in range(self.integ_terms):
            self.add_to_permutation(ndim, self.gauss_order, idx)
            counter += 1
            for idim in range(ndim):
                self.weight[counter] *= weight[idx[idim]]
                self.gpoint[counter][idim] = gpoint[idx[idim]]

    def add_to_permutation(self, nidx, nmax, idx):
        pos = nidx-1
        added_to_permutation = False
        while not added_to_permutation:
            if idx[pos] < nmax-1:
                idx[pos] += 1
                added_to_permutation = True
            else:
                idx[pos] = 0
                pos -= 1
                if pos == -1:
                    raise ValueError('** In add_to_permutation: Attempted to add over nmax on last element **')

    def set_bounds(self, lower, upper):
        self.lower = lower
        self.upper = upper
        if self.integ_terms > 0:
            for igauss in range(self.integ_terms):
                for idim in range(self.ndim):
                    self.weight[igauss] *= 0.5*(upper-lower)
                    self.gpoint[igauss][idim] = 0.5*(upper-lower)*self.gpoint[igauss][idim] + 0.5*(lower+upper)
