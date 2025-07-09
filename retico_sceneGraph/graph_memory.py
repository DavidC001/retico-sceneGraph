"""
A module for memory management of scene graphs and embbedding the triplets found in the scene graph.
For now the memory module is intended to be used with HF Agents by defining tools that can access the scene graph and its bounding boxes.
"""
from copy import deepcopy
import torch
from .incremental_units import SceneGraphUnit
import retico_core
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

from collections import deque
import threading
# import semaphore
import time

import networkx as nx

class SceneGraphEmbedder():
    """
    A class for embedding triplets in a scene graph using a pre-trained model.
    This class uses the SentenceTransformer library to generate embeddings for triplets in a scene graph.
    It can also generate embeddings for queries.
    """

    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", embeddings_precision=None):
        """
        Initializes the SceneGraphEmbedder with a pre-trained model for embedding triplets.
        Parameters:
        - model_name: The name of the pre-trained model to use for embedding.
        - embeddings_precision: The precision of the embeddings. Options are “float32”, “int8”, “uint8”, “binary”, “ubinary”. Default is None (no quantization).
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embedding_precision = embeddings_precision

        self.model = SentenceTransformer(
            self.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def embed_graph(self, graph: nx.Graph):
        """
        Loops through the triplets in the scene graph unit and generates embeddings for them.
        """
        triplets = self.generate_triplets(graph)
        embeddings = self._generate_embeddings(triplets)
        
        embeddings = quantize_embeddings(embeddings, self.embedding_precision) if self.embedding_precision else embeddings
        
        return embeddings
        
    def embed_query(self, query: str) -> torch.Tensor:
        """
        Generates an embedding for a given query string.
        """
        return self.model.encode([query],  prompt_name="query")[0]

    @staticmethod
    def generate_triplets(graph: nx.Graph) -> list[str]:
        """
        Generate sentences from the triplets in the scene graph.
        
        Parameters:
        - graph: A networkx graph representing the scene graph.
        """
        triplets = []
        
        nodes = list(graph.nodes(data=True))
        edges = list(graph.edges(data=True))
        
        for edge in edges:
            source, target, data = edge
            source_label = graph.nodes[source]['label']
            target_label = graph.nodes[target]['label']
            relation = data['label']
            triplets.append(f"{source_label} {source} {relation} {target_label} {target}")
        
        return triplets

    def _generate_embeddings(self, triplets: list[str]):
        """
        Generates embeddings for the triplets in the scene graph unit.
        
        Parameters:
        - triplets: A list of triplets, where each triplet is a tuple of (subject, relation, object).
        """
        document_embeddings = self.model.encode(triplets)
        return document_embeddings

class SceneGraphMemory(retico_core.AbstractConsumingModule):
    """
    A class for managing the memory of a scene graph and the RAG (Retrieval-Augmented Generation) system.
    Note that it will use the "camera_name" metadata field of the Module that produced the ImageIU in input to the SceneGraphModule to identify the source of the image.
    """
    @staticmethod
    def name():
        return "SceneGraphMemory"
    
    @staticmethod
    def description():
        return "A module for managing the memory of a scene graph and the RAG system. It embeds triplets in the scene graph and provides access to the embeddings."
    
    @staticmethod
    def input_ius():
        return [SceneGraphUnit]
    
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", embeddings_precision=None, timeout=0.5, **kwargs):
        """
        Initializes the SceneGraphMemory with a pre-trained model for embedding triplets.
        
        Parameters:
        - model_name: The name of the pre-trained model to use for embedding.
        - embeddings_precision: The precision of the embeddings. Options are “float32”, “int8”, “uint8”, “binary”, “ubinary”. Default is None (no quantization).
        - timeout: The timeout for the background thread to wait for new input. Default is 0.5 seconds.
        """
        super().__init__(**kwargs)
        
        self.embedder_config = {
            "model_name": model_name,
            "embeddings_precision": embeddings_precision
        }
        self.embedder = None
        self._timeout = timeout
        self._is_running = False
        self._input_queue = deque()
        
        # Safe parallel access to the graph variables
        self._lock = threading.Lock()
        
        self._embeddings = {}
        self._graph = {}
        
    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            self._input_queue.append(iu)
        
    def prepare_run(self):
        self.embedder = SceneGraphEmbedder(**self.embedder_config)
        self._is_running = True
        
        # Start the background thread for processing images
        self._memory_thread = threading.Thread(target=self.__memory_loop, daemon=True)
        self._memory_thread.start()
        
    def shutdown(self):
        self._is_running = False
        self._memory_thread.join()
        
    def __memory_loop(self):
        while self._is_running:
            if len(self._input_queue) == 0:
                time.sleep(self._timeout)
                continue
            
            input_iu : SceneGraphUnit = self._input_queue.popleft()
            
            source = input_iu.grounded_in.creator.meta_data.get("camera_name", "unknown")
            
            graph : nx.Graph = input_iu.graph
            new_embeddings = self.embedder.embed_graph(graph)
            
            with self._lock:
                # Store the embeddings and graph in the memory
                self._embeddings[source] = new_embeddings
                self._graph[source] = graph
    
    def get_camera_names(self):
        with self._lock:
            return list(self._graph.keys())
    
    def get_scene_graph(self, camera_name: str) -> nx.Graph:
        """
        Returns the scene graph for a given camera name.
        """
        with self._lock:
            return self._graph.get(camera_name, None)

    def query_camera(self, camera_name: str, query: str, topk=1) -> nx.Graph:
        """
        Queries the memory for a given camera name and returns the scene graph that matches the query and its neighbors.
        """
        with self._lock:
            graph :nx.Graph = deepcopy(self._graph.get(camera_name, None))
            embeddings = torch.tensor(deepcopy(self._embeddings.get(camera_name, None)))

        if graph is None:
            return {"error": "Camera not found"}
        # Perform the query on the graph
        query = torch.tensor(self.embedder.embed_query(query))

        # Get the similarity scores between the query embedding and the graph embeddings
        similarity_scores = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0), embeddings, dim=1
        )
        
        # Get the indices of the top-k most similar embeddings
        topk = min(topk, len(similarity_scores))
        topk_indices = torch.topk(similarity_scores, k=topk).indices
        
        # Construct the graph with only the top-k nodes and their neighbors
        topk_graph = nx.Graph()
        for idx in topk_indices:
            node = graph.nodes[idx.item()]
            topk_graph.add_node(idx.item(), **node)
            for neighbor in graph.neighbors(idx.item()):
                if neighbor not in topk_graph:
                    topk_graph.add_node(neighbor, **graph.nodes[neighbor])
                topk_graph.add_edge(idx.item(), neighbor, **graph.edges[idx.item(), neighbor])
                
        return topk_graph

    def query_memory(self, query: str, topk=1) -> dict[str, nx.Graph]:
        """
        Queries the memory for a given query string and returns the scene graph that matches the query and its neighbors.
        """
        result = {}
        for camera_name in self.get_camera_names():
            graph = self.query_camera(camera_name, query, topk)
            result[camera_name] = graph
        return result
