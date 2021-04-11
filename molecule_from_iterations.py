import numpy as np
from PIL import Image
from reality import atoms, radii, gaps
from scipy.optimize import minimize

class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds
        self.coords = np.array([100 * np.random.random(3) for atom in self.atoms])
        self.iterate_coords()

    def logistic(self, x, max_mag, sig):
        return 2 * max_mag / (1 + np.exp(-x * sig)) - max_mag

    def length(self, v):
        return np.sqrt(np.sum(v**2))

    def force(self, bond):
        """nudge atoms to the correct distance from their neighbours 
        as dictated by self.bonds"""
        ij = list(bond[0])
        ab = [self.atoms[i] for i in ij]
        ab.sort(key=lambda x: atoms.index(x))
        gap = gaps[(ab[0], ab[1], bond[1])]
        dist = self.length(self.coords[ij[0]] - self.coords[ij[1]])
        return (
            abs(self.logistic(gap - dist, max_mag=1, sig=1)) * # close convergence
            self.logistic(gap - dist, max_mag=10, sig=1/10)
        )

    def nudge(self, atom):
        """nudge atoms into straight lines and onto the x-y plane"""
        others = set(range(len(self.atoms))) - self.neighbours(atom, 1) - {atom}
        #print(atom, others)
        diffs = [self.coords[atom] - self.coords[other] for other in others]
        dists = [self.length(diff) for diff in diffs]
        #print(diffs, dists)
        nudges = [
            self.logistic(1/dist, max_mag = 1, sig=10_000) * # govern dropoff
            self.logistic(1/dist, max_mag = 5, sig=100) * diff / dist # govern close
            for dist, diff in zip(dists, diffs)
        ]
        #print(f"{nudges=}")
        #nudge = np.sum(nudges)
        nudge = np.mean(nudges + [np.array((0, 0, 0))], axis=0)
        to_z0 = np.array((0, 0, 0)) # np.array((0, 0, -1)) * self.logistic(self.coords[atom][2], max_mag = 5, sig=0.05)
        #print(f"{to_z0=}")
        return nudge + to_z0 + np.random.random(3) / 100

    def forces(self):
        f = np.array([(0, 0, 0) for atom in self.atoms], dtype="float64")
        for bond in self.bonds:
            mag = self.force(bond)
            obond = list(bond[0])
            vec = (self.coords[obond[1]] - self.coords[obond[0]])
            vec = vec / np.sqrt(np.sum(vec**2))
            f[obond[0]] -= vec * mag
            f[obond[1]] += vec * mag
        for atidx in range(len(self.atoms)):
            f[atidx] += self.nudge(atidx)
        #print("forcemag", self.length(f))
        #print(f"{f=}")
        return f

    def iterate_coords(self, its=1000):
        for i in range(its):
            #print(self.coords)
            self.coords += self.forces()
            #print(np.sqrt(np.sum((self.coords[0] - self.coords[1])**2)))
            #print(np.sqrt(np.sum((self.coords[2] - self.coords[1])**2)))
            #print(np.sqrt(np.sum((self.coords[2] - self.coords[3])**2)))

    def neighbours(self, atom, n):
        """get all atoms <= n steps from atom"""
        nbh = {atom}
        bonds = [b[0] for b in self.bonds]
        for i in range(n):
            nbh = nbh.union(*[b for b in bonds if nbh.intersection(b)])
        return nbh - {atom}

    def as_image(self):
        im = np.zeros((50, 50, 3))
        mi, ma = np.min(self.coords), np.max(self.coords)
        miz, maz = np.min(self.coords, axis=0)[2], np.max(self.coords, axis=0)[2]
        dif = 127/len(self.coords)
        for i, c in enumerate(self.coords):
            im[
                int(49 * (c[0] - mi) / (ma - mi)),
                int(49 * (c[1] - mi) / (ma - mi)),
                :
            ] = (int(255 * (c[2]-miz) / (maz - miz)), 255 - i * dif, 255 - i * dif)
        
        return Image.fromarray(im.astype("uint8"))

def test_molicule():
    h2 = Molecule(["H", "H"], [({0, 1}, 1)])
    assert np.abs(h2.length(h2.coords[0] - h2.coords[1]) - 74) < 1
    
    # check straight line
    h3 = Molecule(["H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1)])
    assert np.abs(h3.length(h3.coords[0] - h3.coords[1]) - 74) < 1
    assert np.abs(h3.length(h3.coords[2] - h3.coords[1]) - 74) < 1
    assert h3.length(h3.coords[0] - h3.coords[2]) > 73 * 2
    
    # check equilateral triangle
    h3c = Molecule(["H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({2, 0}, 1)])
    assert np.abs(h3c.length(h3c.coords[0] - h3c.coords[1]) - 74) < 1
    assert np.abs(h3c.length(h3c.coords[2] - h3c.coords[1]) - 74) < 1
    assert np.abs(h3c.length(h3c.coords[2] - h3c.coords[0]) - 74) < 1
    
    h4 = Molecule(["H", "H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({2, 3}, 1)])
    assert np.abs(h4.length(h4.coords[0] - h4.coords[1]) - 74) < 2
    assert np.abs(h4.length(h4.coords[2] - h4.coords[1]) - 74) < 2
    assert np.abs(h4.length(h4.coords[2] - h4.coords[3]) - 74) < 2
    assert h4.length(h4.coords[0] - h4.coords[3]) > 2 * 74
    assert h4.length(h4.coords[1] - h4.coords[3]) > 74
    #print(Molecule(["H", "H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({2, 0}, 1), ({2, 3}, 1)]).coords)#
    #print(Molecule(["H", "H", "H", "H", "H"], [{0, 1}, {1, 2}, {2, 3}, {1, 4}]).paths)


benzine = Molecule(
    ["C"]*6 + ["H"]*6,
    [
        ({0, 1}, 1),
        ({1, 2}, 2),
        ({2, 3}, 1),
        ({3, 4}, 2),
        ({4, 5}, 1),
        ({5, 0}, 2),

        ({0, 6}, 1),
        ({1, 7}, 1),
        ({2, 8}, 1),
        ({3, 9}, 1),
        ({4, 10}, 1),
        ({5, 11}, 1),
    ],
)


h33 = Molecule(
    ["H"]*6,
    [
        ({0, 1}, 1),
        ({2, 1}, 1),
        ({0, 2}, 1),
        
        ({0, 3}, 1),
        ({1, 4}, 1),
        ({2, 5}, 1),
    ]
)

h4 = Molecule(["H", "H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({2, 3}, 1), ({0, 3}, 1)])
#h3 = Molecule(["H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1)])
h5 = Molecule(["H", "H", "H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({2, 3}, 1), ({3, 4}, 1), ({4, 0}, 1)])
benzine.as_image().save("tempout.png")

if __name__ == "__main__":
    test_molicule()
