"""Got as far as Benzine and decided this "exact" approach was becoming unwieldy."""

import numpy as np
from PIL import Image
from reality import atoms, radii, gaps
from scipy.optimize import minimize

class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds
        print(self.decompose())
        self.set_coords()
        #self.set_coords_from_pathwalk()

    def decompose(self):
        """Gets a minimal decomposition of cycles and other paths."""
        cycles = []
        set_cycles = []
        noncycles = []
        def step_on(path):
            for atom in self.neighbours(path[-1]).intersection(path[:-2]):
                set_cycle = set(path[path.index(atom):])
                if set_cycle not in set_cycles:
                    #print(f"{path=}")
                    #print(f"{atom=}")
                    set_cycles.append(set_cycle)
                    cycles.append(path[path.index(atom):])
                else: # only time we don't explore further is when we are not the first to discover a cycle (previous guy will have gone on)
                    return
            for atom in self.neighbours(path[-1]) - set(path):
                step_on(path + [atom])
        
        for n in self.neighbours(0):
            step_on([0, n])
        
        cycle_atoms = set().union(*set_cycles)
        if not cycle_atoms:
            cycle_atoms = {0} # work with trivial cycle
            
        def walk_path(path):
            for atom in self.neighbours(path[-1]) - {path[-2]}:
                if atom in cycle_atoms: # reached an isolated cycle
                    a_path = path + [atom]
                    if a_path[::-1] not in noncycles:
                        noncycles.append(a_path)
                    return
                walk_path(path + [atom])
            if not self.neighbours(path[-1]) - {path[-2]}:
                noncycles.append(path)
        for atom in cycle_atoms:
            for n in  self.neighbours(atom) - cycle_atoms:
                walk_path([atom, n])
        
        """At this point should have a bunch of distinct cycle sets and paths leading
        between them or to nowhere. It remains to find the minimal cycles and
        maximal paths..."""
        toremove = []
        for cycle in set_cycles:
            for other in [c for c in set_cycles if c != cycle]:
                if other.issubset(cycle): # only care about not sharing one edge
                    toremove.append(cycle)
                    break

        for c in toremove:
            set_cycles.remove(c)
            for list_cycle in cycles:
                if set(list_cycle) == c:
                    cycles.remove(list_cycle)
        #print(noncycles)
        clipped_paths = []
        path_points = set()
        for i in range(len(self.atoms)):
            allpaths_i = [p for p in noncycles if p[0] == i]
            if not allpaths_i:
                continue
            allpaths_i.sort(key=lambda p: len(p))
            clipped_paths.append(allpaths_i[0])
            path_points = path_points.union(set(allpaths_i[0]))
            for j in range(1, len(allpaths_i)):
                cutoff = max((0 if p not in allpaths_i else allpaths_i.index(p)) for p in path_points)
                clipped_paths.append(allpaths_i[j])
                path_points = path_points.union(set(allpaths_i[j]))
        
        return cycles, clipped_paths
                
    def neighbours(self, atomidx):
        #print(atomidx * "-")
        return set().union(*[b[0] for b in self.bonds if atomidx in b[0]]) - {atomidx}            

    def length(self, v):
        return np.sqrt(np.sum(v**2))

    def set_coords(self):
        self.coords = [None for atom in self.atoms]
        cycles, noncycles = self.decompose()
        set_points = set()
        # privelaged starting cycle
        c0 = cycles[0]
        steps0 = [self.gap(*b) for b in zip(c0, c0[1:] + c0[:1])]
        coords0 = self.cyclic_polygon(steps0, 0, (1, 0, 0), (0, 1, 0))
        for atom, coord in zip(c0, coords0):
            self.coords[atom] = coord
            set_points.add(atom)
        while set(range(len(self.atoms))) - set_points:
            #print(f"{set_points=}")
            #print(f"{cycles=}")
            #print(f"{noncycles=}")
            for c in cycles[1:]:
                anchor = set(c).intersection(set_points)
                if anchor:
                    if len(anchor) != 2:
                        raise Exception("not implemented for cycles with weird joins")
                    steps = [self.gap(*b) for b in zip(c, c[1:] + c[:1])]
                    oanchor = list(anchor)
                    c = c[::-1] # todo: deal with knowing inside from outside...
                    anch_ids = [c.index(a) for a in oanchor]
                    anch_ids.sort()
                    x_axis = self.coords[c[anch_ids[1]]] - self.coords[c[anch_ids[0]]]
                    x_axis = x_axis / self.length(x_axis)
                    #print(c, c[anch_ids[1]], c[anch_ids[0]])
                    #print(self.coords[c[anch_ids[1]]], self.coords[c[anch_ids[0]]])
                    #print(x_axis)
                    y_axis = np.cross(x_axis, (0, 0, -1))
                    #print(y_axis) # this looks goods so far
                    # need to construct cyclic polygon with the right orientation...
                    coords = self.cyclic_polygon(steps, anch_ids[0], x_axis, y_axis)
                    diff = self.coords[c[anch_ids[0]]] - coords[anch_ids[0]]
                    coords = [coord + diff for coord in coords]
                    #print(f"{c=}, {coords=}, {steps=}")
                    for atom, coord in zip(c, coords):
                        self.coords[atom] = coord
                        set_points.add(atom)

            for p in noncycles:
                anchor = set(p).intersection(set_points)
                if anchor:
                    anchor = max(p.index(a) for a in anchor)
                    p = p[anchor:] # anchor is p[0]
                    print(p)
                    ns = self.neighbours(p[0]).intersection(set_points)
                    print(ns)
                    d = np.sum([v / self.length(v) for v in [self.coords[p[0]] - self.coords[n] for n in ns]], axis=0)
                    print("dirs :", [v / self.length(v) for v in [self.coords[p[0]] - self.coords[n] for n in ns]])
                    if self.length(d) < 1/100:
                        print("branching")
                        d = np.cross(self.coords[ns.pop()] - self.coords[p[0]], (0, 0, 1))
                    d = d / self.length(d)
                    print(d)
                    steps = [self.gap(*b) for b in zip(p[:-1], p[1:])]
                    coords = [np.sum(steps[:k]) * d for k in range(len(steps) + 1)]
                    print(f"{coords=}")
                    diff = self.coords[p[0]] - coords[0]
                    coords = [coord + diff for coord in coords]
                    for atom, coord in zip(p, coords):
                        self.coords[atom] = coord
                        print(f"setting {atom} to {coord}")
                        set_points.add(atom)


    """
    def set_coords_from_pathwalk(self):
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

        cycles, noncycles = self.decompose(paths)
                    
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
    """

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
        angles = np.array(angles[start:] + angles[:start]) - angles[start + 1]/2 # side from start to start+1 should be on x_axis
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
        im = np.zeros((50, 50, 3))
        mi, ma = np.min(self.coords), np.max(self.coords)
        miz, maz = np.min(self.coords, axis=0)[2], np.max(self.coords, axis=0)[2]
        dif = 127/len(self.coords)
        for i, c in enumerate(self.coords):
            im[
                int(49 * (c[0] - mi) / (ma - mi)),
                int(49 * (c[1] - mi) / (ma - mi)),
                :
            ] = (int(255 * (c[2]-miz) / (maz - miz + 0.01)), 255 - i * dif, 255 - i * dif)
        
        return Image.fromarray(im.astype("uint8"))

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

caffeine = Molecule(
    ["C"]*3 + ["N", "C", "N"] + # hexagon
    ["N", "C", "N"] + # pentagon
    ["O"] * 2 +
    ["C", "H", "H", "H"] * 3 +
    ["H"],
    [
        ({0, 1}, 1), # hexagon
        ({1, 2}, 2),
        ({2, 3}, 1),
        ({3, 4}, 1),
        ({4, 5}, 1),
        ({5, 0}, 1),
        
        ({1, 6}, 1), # pentagon
        ({6, 7}, 1),
        ({7, 8}, 2),
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

#h2 = Molecule(["H", "H", "H"], [({0, 1}, 1), ({1, 2}, 1), ({0, 2}, 1)])

caffeine.as_image().show()

if __name__ == "__main__":
    test_molicule()
