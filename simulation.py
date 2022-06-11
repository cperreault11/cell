"""
Main module that defines a simulation class
"""
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from cell import Cell


class Simulation():
    """
    Represents an instance of cell simulation

    Dataset structure:
    cell ID: [(timestep, position, action)]
    cell ID: [(timestep, neighbors, action)]

    I think that's all I need for the simplest form. Obviously can track lineage for more info, but we're not looking
    at that right now at all

    Get data the same way as the other paper: just middle .65 square

    This will work for the likelihood simple function. Not sure about the point processes though-- I think it will be fine though.
    """

    def __init__(self, L, num_cells):
        """
        Initialize a new simulation with an LxL grid and a certain number of initial cells
        """
        self.L = L
        self.num_cells = num_cells
        self.cells = [Cell(np.random.rand(2) * L) for i in range(num_cells)]
        self.p = 0.01
        
    def within_range(self, cell, ratio=0.65):
        """
        Check if a cell is within the center of the simulation, to match paper
        """
        ratio = 1- ratio
        return all(cell.pos > np.array([ratio/2, ratio/2])) & all(cell.pos < np.array([self.L - ratio/2, self.L-ratio/2]))

    def step(self,actions,i,timestep=1.2, sample=False):
        """
        Make a single step in the simulation without any delaminations or divides
        """
        copy = np.array([cell.pos for cell in self.cells])
        to_remove = set()
        to_add = set()
        deleted, divided = 0,0
        for cell in self.cells:
            cell.update_pos(copy, timestep, self.L)
        if sample:
            snapshot_position = [[i,cell.id, cell.pos.copy()[0], cell.pos.copy()[1], str(cell.next)if cell.lifetime is not None and cell.lifetime <= 1e-6 else "none", cell.parent, cell.lifetime if cell.lifetime is not None and cell.lifetime <= 1e-6 else "none"] for cell in self.cells if self.within_range(cell)]
            #return np.array([snapshot_position],dtype=object).T
            snapshot_position = pd.DataFrame(np.array(snapshot_position,dtype=object), columns=['i','id','x','y','next','parent','lifetime'])
            #graph = self.get_graph()
            #snapshot_graph = [[cell.id, str(cell.next) if cell.lifetime is not None and cell.lifetime <= 1e-6 else None, graph[1][graph[0][i]:graph[0][i+1]]] for i, cell in enumerate(self.cells) if self.within_range(cell)]
        for cell in self.cells:
            if cell.lifetime is not None and cell.lifetime <= 1e-6:
                if cell.next == "del":
                    to_remove.add(cell)
                    deleted += 1
                    if not cell.follow:
                        available = [cell for cell in self.cells if cell.next is None and cell not in to_remove]
                        positions = np.array([cell.pos for cell in available])
                        distances = cell.distance(positions, np.array([self.L,self.L]))
                        ind = np.random.choice(np.argpartition(distances, 6)[:6])
                        available[ind].set_divide(np.random.randint(0,7)* 1.2 + 44.4, True,cell.id)
                    cell.delaminate()
                elif cell.next == "div":
                    divided+=1
                    to_add.add(Cell(np.random.random(2) * .002 * self.L - .001 * self.L + cell.pos))
                    if not cell.follow:
                        available = [cell for cell in self.cells if cell.next is None and cell not in to_remove]
                        positions = np.array([cell.pos for cell in available])
                        distances = cell.distance(positions, np.array([self.L,self.L]))
                        ind = np.random.choice(np.argpartition(distances, 6)[:6])
                        available[ind].set_delaminate(np.random.randint(0,7)* 1.2 + 44.4, True, cell.id)
                    cell.divide()
            if actions == "del" and cell.next is None:
                r = np.random.random()
                if r < self.p:
                    cell.set_delaminate(np.random.randint(0,7)* 1.2 + 32.4, False, "none")
            elif actions == "div" and cell.next is None:
                r = np.random.random()
                if r < self.p:
                    cell.set_divide(np.random.randint(0,7) * 1.2 + 32.4, False, "none")
            elif actions == "both" and cell.next is None:
                r = np.random.random()
                if r < self.p / 2:
                    cell.set_delaminate(np.random.randint(0,7)* 1.2 + 32.4, False, "none")
                elif r < self.p:
                    cell.set_divide(np.random.randint(0,7) * 1.2 + 32.4, False,"none")
        for cell in to_add:
            self.cells.append(cell)
        for cell in to_remove:
            self.cells.remove(cell)
        if sample:
            return snapshot_position#, snapshot_graph, deleted, divided, len(self.cells)

    def run(self, actions, timestep, steps, sample = False):
        """
        Run a simulation for a certain number of timesteps

        use the actions provided by the input object
        """
        #results = pd.DataFrame()
        positions = []
        temp = []
        #deleted, divided, num_cells = [],[],[]
        for i in range (steps):
            if sample:
                temp.append(self.step(actions, i,timestep = timestep, sample = True))
            else:
                self.step(actions, i, timestep = timestep)
            if sample and  i % 20 == 0:
                # combine to a data row
                df = pd.concat(temp)
                pos = df.sort_values('i').groupby('id').last()[['i','x','y']].reset_index()
                fate_list = []
                parent_list = []
                for id in pos['id']:
                    fate = df[(df['id'] ==id)&(df['lifetime'] != "none")]
                    fate = fate[fate['lifetime'] < 1e-6]
                    if len(fate) > 0:
                        fate_list.append(fate.reset_index().at[0,'next'])
                        parent_list.append(fate.reset_index().at[0,'parent'])
                    else:
                        fate_list.append("none")
                        parent_list.append("none") 
                pos.insert(0, "next", fate_list)
                pos.insert(0,'parent', parent_list)
                pos.loc[pos['next'] == "del",'i'] = i
                #pos[pos['next'] != "none"]['i'] = i # may just want to track all of them that start in the frame
                positions.append(pos)
                temp = []
        if sample:
            return pd.concat(positions)

    def get_graph(self):
        """
        Get the graph of points using Voronoi tesselation
        """
        points = np.array([cell.pos for cell in self.cells])
        tri = Delaunay(points)
        return tri.vertex_neighbor_vertices

