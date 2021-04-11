import numpy as np
from PIL import Image
from reality import atoms, radii, gaps
from write_tube import tube, save_stl, sphere

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
    
    def dist(self, a, b):
        return self.length(self.coords[a] - self.coords[b])

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
        diffs = [self.coords[atom] - self.coords[other] for other in others]
        dists = [self.length(diff) for diff in diffs]
        nudges = [
            self.logistic(1/dist, max_mag = 1, sig=10_000) * # govern dropoff
            self.logistic(1/dist, max_mag = 5, sig=100) * diff / dist # govern close
            for dist, diff in zip(dists, diffs)
        ]
        nudge = np.mean(nudges + [np.array((0, 0, 0))], axis=0)
        to_z0 = np.array((0, 0, 0))
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
        return f

    def iterate_coords(self, its=1000):
        for i in range(its):
            self.coords += self.forces()

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

    def to_stl(self, name, scale=1/10):
        """one picometer gets scaled to `scale` millimeters"""
        triangles = []
        for bond in self.bonds:
            obond = list(bond[0])
            triangles += tube(
                [
                    self.coords[obond[0]] * scale,
                    self.coords[obond[1]] * scale,
                ],
                r=1, # todo : depend on strength of bond?
                k=20,
                loop=False,
            )
        for i, atom in enumerate(self.atoms):
            triangles += sphere(
                self.coords[i] * scale,
                radii[atom] * scale,
                20,
                20,
            )
        save_stl(triangles, name)

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
        ({0, 1}, 1.5),
        ({1, 2}, 1.5),
        ({2, 3}, 1.5),
        ({3, 4}, 1.5),
        ({4, 5}, 1.5),
        ({5, 0}, 1.5),

        ({0, 6}, 1),
        ({1, 7}, 1),
        ({2, 8}, 1),
        ({3, 9}, 1),
        ({4, 10}, 1),
        ({5, 11}, 1),
    ],
)

caffeine = Molecule(
    ["C"]*3 + ["N", "C", "N"] + # hexagon
    ["N", "C", "N"] + # pentagon
    ["O"] * 2 +
    ["C", "H", "H", "H"] * 3 +
    ["H"]
    [
        ({0, 1}, 1), # hexagon
        ({1, 2}, 2),
        ({2, 3}, 1),
        ({3, 4}, 1),
        ({4, 5}, 1),
        ({5, 0}, 1),
        
        ({1, 6}, 1), # pentagon
        ({6, 7}, 1),
        ({7, 8}, 2), # todo this electron is shared (?)
        ({8, 2}, 1),
        
        ({0, 9}, 2), # oxygens
        ({4, 10}, 2),
        
        ({5, 11}, 1), # methyl group
        ({11, 12}, 1),
        ({11, 13}, 1),
        ({11, 14}, 1),
        
        ({3, 15}, 1), # methyl group
        ({15, 16}, 1),
        ({15, 17}, 1),
        ({15, 18}, 1),

        ({6, 19}, 1), # methyl group
        ({19, 20}, 1),
        ({19, 21}, 1),
        ({19, 22}, 1),
        
        ({7, 23}, 1),
    
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
benzine.iterate_coords(2000)
benzine.as_image().save("tempout.png")
benzine.to_stl("benzine")

if __name__ == "__main__":
    test_molicule()
