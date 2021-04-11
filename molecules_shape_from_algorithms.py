"""Got as far as Benzine and decided this "exact" approach was becoming unwieldy."""

import numpy as np
from PIL import Image
from reality import atoms, radii, gaps
from scipy.optimize import minimize

class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds
        # decompose by walking the graph
        paths = []
        bonds_left = [set(b[0]) for b in self.bonds]
        while bonds_left:
            path = list(bonds_left[0])[:1]
            for _ in (0, 1): # extend path in both directions
                if path[-1] not in set().union(*bonds_left):
                    break
                bond = next(b for b in bonds_left if path[-1] in b)
                bonds_left.remove(bond)
                end = (bond - {path[-1]}).pop()
                while end in set().union(*bonds_left):
                    print(path, bonds_left, end)
                    bond = next(b for b in bonds_left if end in b)
                    path.append(end)
                    bonds_left.remove(bond)
                    end = (bond - {path[-1]}).pop()
                    if end in path:
                        cycle = path[path.index(end):] + [end]
                        paths.append(cycle)
                        path = path[:path.index(end)]
                        break
                if not path:
                    break
                path.append(end)
                path = path[::-1]
            paths.append(path)
        paths = [p for p in paths if p]
        print(paths)
        self.coords = [None for atom in self.atoms]
        path = paths[0]
        steps = [self.gap(*b) for b in zip(path[:-1], path[1:])]
        if path[0] == path[-1]: # cycle 
            coords = self.cyclic_polygon(steps, 0, (1, 0, 0), (0, 1, 0))
        else:
            coords = [np.array((1, 0, 0)) * sum(steps[:k]) for k in range(len(steps) + 1)]
        for atom, coord in zip(paths[0], coords):
            self.coords[atom] = coord
        set_coords = set(paths[0])
        paths = paths[1:]
        if not paths:
            return
        print(self.coords)
        while paths:
            for path in paths:
                if set(path).intersection(set_coords):
                    atom = set(path).intersection(set_coords).pop()
                    neighbours = set().union(*[set(b[0]) for b in self.bonds if atom in set(b[0])]) - {atom}
                    neighbours = neighbours.intersection(set_coords)
                    radial_vector = np.mean([self.coords[atom] - self.coords[n] for n in neighbours], axis=0)
                    tangent =  self.coords[neighbours.pop()] - self.coords[atom]
                    tangent = tangent / np.sqrt(np.sum(tangent**2))
                    if np.mean(radial_vector**2) < 1: # in case of attatching to straight line
                        radial_vector = np.cross((0, 0, 1), tangent)
                    else:
                        radial_vector = radial_vector / np.sqrt(np.sum(radial_vector**2))
                    print("r : ", radial_vector, tangent)
                    y_axis = np.cross(radial_vector, tangent)
                    steps = [self.gap(*b) for b in zip(path[:-1], path[1:])]
                    print("STEPS", steps)
                    if path[0] == path[-1]: # cycle 
                        coords = self.cyclic_polygon(steps, path.index(atom), radial_vector, y_axis)
                    else:
                        coords = [(-1 if path[-1] == atom else 1) * radial_vector * (sum(steps[:k])) for k in range(len(steps) + 1)]
                    
                    diff = self.coords[atom] - coords[path.index(atom)]
                    coords = [coord + diff for coord in coords]
                    for atom, coord in zip(path, coords):
                        self.coords[atom] = coord
                    set_coords = set_coords.union(path)
                    paths.remove(path)
                    break
        

    def rotate(self, points, angle, axis=(0, 0, 1)):
        mat = [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
        return [np.matmul(mat, p) for p in points]

    def cyclic_polygon(self, steps, start, x_axis, y_axis):
        def angle_err(theta):
            r = (max(steps)/2) / np.sin(theta/2)
            return np.abs(sum(
                np.arccos(1 - step**2/(2* r**2))
                for step in steps
            ) - 2 * np.pi)
        max_angle = minimize(
            angle_err,
            2*np.pi / len(steps),
        )
        r = (max(steps)/2) / np.sin(max_angle.x[0]/2)
        angles = [
            sum(
                np.arccos(1 - step**2/(2* r**2))
                for step in steps[:k]
            )
            for k in range(len(steps))
        ]
        angles = angles[start:] + angles[:start]
        x_axis, y_axis = np.array(x_axis), np.array(y_axis)
        return [
            x_axis + x_axis * r * np.cos(angle) + y_axis * r * np.sin(angle)
            for angle in angles
        ]

    def getbond(self, a, b):
        for bond in self.bonds:
            if set(bond[0]) == {a, b}:
                return bond
        raise Exception(f"no bond between {a} and {b}")

    def gap(self, a, b): # bad
        bond = self.getbond(a, b)
        pair, e = bond
        pair = list(pair)
        pair = [self.atoms[p] for p in pair]
        pair.sort(key=lambda x: atoms.index(x))
        pair = tuple(pair)
        return gaps[(*pair, e)]
    
    def as_image(self):
        im = np.zeros((50, 50))
        mi, ma = np.min(self.coords), np.max(self.coords)
        for c in self.coords:
            im[
                int(49 * (c[1] - mi) / (ma - mi)),
                int(49 * (c[0] - mi) / (ma - mi)),
            ] = 255
        
        return im

def test_molicule():
    pass
    #print(Molecule(["H", "H"], [({0, 1}, 1)]).coords)
    #print(Molecule(["H", "H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({2, 3}, 1)]).coords)
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

Image.fromarray(benzine.as_image()).show()

if __name__ == "__main__":
    test_molicule()
