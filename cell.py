"""
Module to describe a single simulated cell

Author: Conor Perreault
"""
import uuid
import numpy as np

class Cell():
    """
    Represents a single cell that has repulsive forces with the other cells
    """
    l = 0.125
    K = 1/9 # notation feels weird but this seems to fix it, so might be right. Check the units to be sure.s
    def __init__(self, pos):
        """
        Initialize a cell with a given position
        """
        self.pos = pos
        self.id = uuid.uuid1()
        self.lifetime = None
        self.next = None # delaminate or divide
        self.follow = False
        self.parent = "none"

    def set_divide(self, lifetime, follow, parent):
        """
        Tell a cell to divide
        """
        if parent == "none":
            print ("here")
        self.next = "div"
        self.lifetime = lifetime
        self.follow = follow
        self.parent = parent
    
    def divide(self):
        """
        divide and reset
        """
        if self.parent=="none":
            print("here")
        self.next = None
        self.lifetime =  None
        self.follow = False
        self.parent = "none"

    def set_delaminate(self, lifetime, follow,parent):
        """
        Tell a cell to delaminate
        """
        self.next = "del"
        self.lifetime = lifetime
        self.follow = follow
        self.parent = parent

    def delaminate(self):
        """
        delaminate and reset 
        """
        self.next = None
        self.lifetime =  None
        self.follow = False
        self.parent = "none"

    def get_sum_forces(self, neighbors,dimensions):
        """
        Return the forces required to move this cell at the current timestep
        """
        force = sum(self.get_force(neighbor,dimensions) for neighbor in neighbors if neighbor.id != self.id)
        return force

    def get_force(self, neighbor, dimensions):
        """
        Get the repulsive forces between this cell and a single neighbor
        """
        dist = self.distance(neighbor,dimensions)
        if dist > Cell.l:
            return np.zeros(2)
        direction = neighbor.pos - self.pos
        wrap = dist != np.linalg.norm(direction)
        if wrap:
            direction = - direction
        return Cell.K * (dist - Cell.l) * direction / np.linalg.norm(direction) # need to update the direction vector here to account for periodic boundary

    def distance(self, other, dimensions):
        """
        Vectorized distance function
        """
        delta = np.abs(self.pos - other)
        delta = np.where(delta > .5 * dimensions, delta - dimensions, delta)
        return np.hypot(delta[:,0], delta[:,1])

    def get_forces(self, other, dimensions):
        """
        Vectorized force calculation
        """
        distances = self.distance(other, dimensions)
        neighbors = other[(distances < .125) & (distances > 1e-8)]
        distances = distances[(distances < .125) & (distances > 1e-8)]
        directions = self.pos - neighbors
        wrap = np.abs(distances - np.linalg.norm(directions, axis=1)) > 1e-6
        directions[wrap] = -directions[wrap]
        return -(1/9)*(directions.T*(distances - 0.125) / np.linalg.norm(directions, axis=1)).T.sum(axis=0)

    def update_pos(self, neighbors,timestep, L):
        """
        Update the position of a cell
        """
        velocity = self.get_forces(neighbors, np.array([L,L]))
        self.pos += velocity * timestep
        self.pos %= L
        self.pos = np.where((self.pos >= 0) & (self.pos <= L), self.pos, self.pos + L)
        if self.lifetime is not None:
            self.lifetime -= timestep
        return round(self.lifetime,1) if self.lifetime is not None else None