"""
Contains the incremental units for scene graph generation in the Retico framework.
"""

from retico_core.abstract import IncrementalUnit
import networkx as nx

class SceneGraphUnit(IncrementalUnit):
    """
    An abstract class for scene graph units in the Retico framework.
    It contains the graph retrieved from the image in a networkx graph format.
    It also stores the bounding boxes of objects and subjects, as well as the relationships between them.
    """
    @staticmethod
    def type():
        return "SceneGraphUnit"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._graph = nx.Graph()
        self._bboxes = {}
        
    @property
    def graph(self) -> nx.Graph:
        """
        Returns the current scene graph.
        """
        return self._graph
    
    @property
    def bounding_boxes(self) -> dict[int, tuple[int, int, int, int]]:
        """
        Returns the bounding boxes of the nodes in the scene graph.
        """
        return self._bboxes
    
    def set_graph(self, graph):
        """
        Sets the current scene graph.
        """
        self._graph = graph

    def build_graph(self, nodes:list[str], edges:list[tuple[str, tuple[int,int]]]):
        """
        Builds a scene graph from the given nodes and edges.
        
        Parameters:
        - nodes: A list of node labels.
        - edges: A list of edges, where each edge is a tuple containing the edge label and a tuple of source and target node indices.
        """
        
        self._graph.clear()
        for i, node in enumerate(nodes):
            self._graph.add_node(i, label=node)
            
        for edge in edges:
            label, (source, target) = edge
            self._graph.add_edge(source, target, label=label)

    def clear_graph(self):
        """
        Clears the current scene graph.
        """
        self._graph.clear()
        
    def set_bboxes(self, bboxes:dict[int, tuple[int, int, int, int]]):
        """
        Sets the bounding boxes for the nodes in the scene graph.
        Parameters:
        - bboxes: A dictionary where keys are node labels and values are tuples representing the bounding box (x1, y1, x2, y2).
        """
        self._bboxes = bboxes
        
    # to string method
    def __str__(self):
        """
        Returns a string representation of the scene graph.
        """
        return f"SceneGraphUnit(graph={self._graph}, bboxes={len(self._bboxes)})"
        
