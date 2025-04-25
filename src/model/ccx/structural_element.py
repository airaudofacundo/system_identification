import numpy as np

class structural_element:

    def __init__(self):
        pass

    def set_npoin(self, npoin):
        self.npoin = npoin
        self.point = np.zeros(npoin, dtype = np.int32)

    def set_point(self, point):
        self.point = point

    def set_young_modulus(self, young_modulus):
        self.young_modulus = young_modulus

    def set_poisson_coef(self, poisson_coef):
        self.poisson_coef = poisson_coef

    def set_density(self, density):
        self.density = density

    def compute_centroid(self, point_coord):
        xc = 0.0
        yc = 0.0
        zc = 0.0
        for ipoin in self.point:
            xc += point_coord[ipoin][0]
            yc += point_coord[ipoin][1]
            zc += point_coord[ipoin][2]
        xc /= self.npoin
        yc /= self.npoin
        zc /= self.npoin
        return xc, yc, zc
