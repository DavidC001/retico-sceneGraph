"""
Retico Scene Graph Module
This module provides functionality for scene graph generation from images using a pre-trained RelTR model.
It processes images to extract objects, their relationships, and generates a scene graph.
It also includes a drawing module to visualize the generated scene graphs on images.
"""

import time
import retico_core
import retico_vision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import box_iou
from PIL import Image, ImageDraw
from .incremental_units import SceneGraphUnit
from collections import deque
import threading
import networkx as nx

"""packages imported from RelTR repository"""
import sys
import os
sys.path.append(os.getenv("RelTR_PATH", "./RelTR"))
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

class SceneGraphModule(retico_core.AbstractModule):
    """
    A Retico module for scene graph generation.
    """
    classes = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    predicates = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
    
    @staticmethod
    def name():
        return "SceneGraphModule"
    
    @staticmethod
    def description():
        return "A Retico module for scene graph generation from images. It processes images to extract objects, their relationships, and generates a scene graph."
    
    def __init__(self, topk=10, model_path="./checkpoint0149.pth", IoU_threshold=0.5, timeout=0.75, **kwargs):
        """
        Initializes the SceneGraphModule with the given classes and predicates.
        
        Parameters:
        - topk: Number of top predictions to keep (default is 10).
        - model_path: Path to the pre-trained model checkpoint (default is ./checkpoint0149.pth).
        - IoU_threshold: Threshold for Intersection over Union to consider two bounding boxes as the same object (default is 0.75).
        - timeout: Time to wait when no image is available in the processing thread (default is 0.5 seconds).
        """
        super().__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.topk = topk  # Number of top predictions to keep
        self.IoU_threshold = torch.tensor(IoU_threshold, dtype=torch.float32, device=self.device)
        self.model_path = model_path  # Path to the pre-trained model checkpoint
        
        self._image_queue = deque(maxlen=1)  # Queue to hold the latest image IU
        self._timeout = timeout  # Time to wait when no image is available
        self._is_running = False
        
    
    @staticmethod
    def input_ius():
        return [retico_vision.ImageIU]
    
    @staticmethod
    def output_iu():
        return SceneGraphUnit
    
    def prepare_run(self):
        self._is_running = True
        
        self._prepare_model()
        # launch processing thread for image processing
        self._image_thread = threading.Thread(target=self._process_image_loop, daemon=True)
        self._image_thread.start()
    
    def shutdown(self):
        self._is_running = False
        if self._image_thread.is_alive():
            self._image_thread.join()
    
    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            self._image_queue.append(iu)
    
    def _prepare_model(self):
        position_embedding = PositionEmbeddingSine(128, normalize=True)
        backbone = Backbone('resnet50', False, False, False)
        backbone = Joiner(backbone, position_embedding)
        backbone.num_channels = 2048

        transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                normalize_before=False,
                                return_intermediate_dec=True)

        self.model = RelTR(backbone, transformer, 
                           num_classes=151, 
                           num_rel_classes = 51,
                    num_entities=100, num_triplets=200)

        # The checkpoint is pretrained on Visual Genome
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(self.model_path, weights_only=False)
        self.model.load_state_dict(state_dict["model"])
        self.model.eval()
        self.model.to(self.device)
        
        self.transforms = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
        return b

    def _get_model_output(self, image: Image.Image) -> dict:
        """
        Processes the input image through the model to get the output.
        
        Parameters:
        - image: The input image to process.
        
        Returns: 
        - A dictionary containing "sub_bboxes", "obj_bboxes", "sub_logits", "obj_logits", and "rel_logits".
        The bounding boxes are rescaled to the original image size, and logits are filtered based on confidence scores.
        """
        # propagate through the model
        img = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)

        # keep only predictions with >0.3 confidence
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                probas_obj.max(-1).values > 0.3))
        
        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = self.rescale_bboxes(outputs['sub_boxes'][0, keep], image.size)
        obj_bboxes_scaled = self.rescale_bboxes(outputs['obj_boxes'][0, keep], image.size)

        topk = self.topk
        if keep.sum() < topk:
            topk = keep.sum().item()
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]

        keep_queries = keep_queries[indices]
        
        return {
            'sub_bboxes': sub_bboxes_scaled[indices],
            'obj_bboxes': obj_bboxes_scaled[indices],
            "sub_logits": probas_sub[keep_queries],
            "obj_logits": probas_obj[keep_queries],
            "rel_logits": probas[keep_queries],
        }

    def _build_scene_graph(self, output: dict) -> tuple[list[str], list[tuple[str, tuple[int,int], tuple[int,int,int,int]]]]:
        """
        Builds a scene graph from the model output in the format expected by SceneGraphUnit and matches the subjects and objects based on the bounding boxes.
        
        Parameters:
        - output: A dictionary containing the model output with keys 'sub_bboxes', 'obj_bboxes', 'sub_logits', 'obj_logits', and 'rel_logits'.
        
        returns:
        Tuple containing:
            - List[str] of nodes with their labels
            - List[tuple[str, tuple[int,int]]] of edges with their labels and source/target node indices
            - List[(int,int,int,int)] of bounding boxes for subjects and objects
        """
        sub_bboxes = output['sub_bboxes']
        obj_bboxes = output['obj_bboxes']
        sub_logits = output['sub_logits']
        obj_logits = output['obj_logits']
        rel_logits = output['rel_logits']
        
        len_sub = sub_bboxes.shape[0]
        
        # Concatenate bboxes and logits once
        bboxes = torch.cat((sub_bboxes, obj_bboxes), dim=0)
        logits = torch.cat((sub_logits, obj_logits), dim=0)
        
        # Get predicted labels for all boxes at once
        predicted_labels = torch.argmax(logits, dim=1)
        label_names = [self.classes[idx.item()] for idx in predicted_labels]
        
        # Compute IoU matrix once
        iou_matrix = box_iou(bboxes, bboxes)
        
        # Use clustering approach for node matching
        nodes = []
        node_mapping = [-1] * len(bboxes)  # Pre-allocate mapping array
        
        final_bboxes = []
        
        for i in range(len(bboxes)):
            if node_mapping[i] != -1:  # Already assigned to a cluster
                continue
                
            current_label = label_names[i]
            node_idx = len(nodes)
            nodes.append(current_label)
            final_bboxes.append(bboxes[i].tolist())
            node_mapping[i] = node_idx
            
            # Find all matching boxes for this cluster in one pass
            matches = torch.where(
                (iou_matrix[i] > self.IoU_threshold) & 
                torch.tensor([label == current_label for label in label_names], device=self.device)
            )[0]
            
            # Assign all matches to this cluster
            for match_idx in matches:
                if node_mapping[match_idx.item()] == -1:
                    node_mapping[match_idx.item()] = node_idx
            
        
        # Create edges efficiently
        rel_labels = torch.argmax(rel_logits, dim=1)
        edges = [
            (self.predicates[rel_labels[i].item()], (node_mapping[i], node_mapping[i + len_sub]))
            for i in range(len_sub)
        ]
        
        return nodes, edges, final_bboxes
        

    def _process_image_loop(self):
        while self._is_running:
            if len(self._image_queue) == 0:
                time.sleep(self._timeout)
                continue
            
            image_iu = self._image_queue.popleft()
            
            image = image_iu.image
            if image is None:
                continue

            # process the image through the model
            output = self._get_model_output(image)
            # build the scene graph from the model output
            nodes, edges, bboxes = self._build_scene_graph(output)
            
            # create a new SceneGraphUnit with the output
            output_iu : SceneGraphUnit = self.create_iu(image_iu)
            output_iu.build_graph(nodes, edges)
            output_iu.set_bboxes(bboxes)
            
            update_message = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(update_message)
            
            
class SceneGraphDrawingModule(retico_core.AbstractModule):
    """A Retico module for drawing scene graphs onto images."""
    
    @staticmethod
    def name():
        return "SceneGraphDrawingModule"
    
    @staticmethod
    def description():
        return "A Retico module for drawing scene graphs onto images. It takes SceneGraphUnit as input and outputs an ImageIU with the drawn scene graph."
    
    def input_ius(self):
        return [SceneGraphUnit]
    
    def output_iu(self):
        return retico_vision.ImageIU
    
    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue

            if not isinstance(iu, SceneGraphUnit):
                continue

            # Draw the scene graph onto the image
            image = self._draw_scene_graph(iu)

            # Create a new ImageIU with the drawn image
            image_iu = self.create_iu(iu)
            image_iu.image = image
            update_message = retico_core.UpdateMessage.from_iu(image_iu, retico_core.UpdateType.ADD)
            self.append(update_message)

    def _draw_scene_graph(self, scene_graph_iu: SceneGraphUnit) -> Image.Image:
        """
        Draws the scene graph onto the image.
        
        Parameters:
        - scene_graph_iu: The SceneGraphUnit containing the scene graph to draw.
        
        Returns:
        - An PIL Image object with the drawn scene graph.
        """
        grounded_in = scene_graph_iu.grounded_in
        if not isinstance(grounded_in, retico_vision.ImageIU):
            return None
        image = grounded_in.image
        
        #check if bboxes are available
        if len(scene_graph_iu.bounding_boxes) == 0:
            return image
        
        # Draw the bounding boxes and labels on the image
        draw = ImageDraw.Draw(image)
        bboxes = scene_graph_iu.bounding_boxes
        graph : nx.Graph = scene_graph_iu.graph

        nodes = graph.nodes(data=True)
        edges = graph.edges(data=True)

        # Helper function to get bounding box center
        def get_bbox_center(bbox):
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Helper function to draw arrow
        def draw_arrow(draw, start, end, color="blue", width=3):
            # Draw the main line
            draw.line([start, end], fill=color, width=width)
            
            # Calculate arrow head
            import math
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            arrow_length = 15
            arrow_angle = math.pi / 6  # 30 degrees
            
            # Arrow head points
            arrow_x1 = end[0] - arrow_length * math.cos(angle - arrow_angle)
            arrow_y1 = end[1] - arrow_length * math.sin(angle - arrow_angle)
            arrow_x2 = end[0] - arrow_length * math.cos(angle + arrow_angle)
            arrow_y2 = end[1] - arrow_length * math.sin(angle + arrow_angle)
            
            # Draw arrow head
            draw.polygon([(end[0], end[1]), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)], fill=color)
        
        # Helper function to draw text with background
        def draw_text_with_background(draw, position, text, text_color="white", bg_color="blue", font_size=12):
            # Get text bounding box
            bbox = draw.textbbox(position, text)
            padding = 2
            bg_bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
            
            # Draw background rectangle
            draw.rectangle(bg_bbox, fill=bg_color, outline=None)
            # Draw text
            draw.text(position, text, fill=text_color)

        # Draw bounding boxes with improved styling
        for node, bbox in zip(nodes, bboxes):
            # Use different colors for different object types
            bbox_color = "lime" if "person" in node[1]["label"].lower() else "red"
            draw.rectangle(bbox, outline=bbox_color, width=3)
            
            # Draw label with background for better readability
            label_pos = (bbox[0] + 2, bbox[1] - 20)
            draw_text_with_background(draw, label_pos, node[1]["label"], "white", bbox_color)
        
        # Draw edges with improved visualization
        for edge in edges:
            source, target = edge[:2]
            label = edge[2]['label']
            source_bbox = bboxes[source]
            target_bbox = bboxes[target]
            
            # Get centers of bounding boxes
            source_center = get_bbox_center(source_bbox)
            target_center = get_bbox_center(target_bbox)
            
            # Draw arrow from source to target
            draw_arrow(draw, source_center, target_center, "dodgerblue", 3)
            
            # Draw the relationship label at the midpoint with background
            mid_x = (source_center[0] + target_center[0]) / 2
            mid_y = (source_center[1] + target_center[1]) / 2
            draw_text_with_background(draw, (mid_x, mid_y), label, "white", "dodgerblue")
                

        return image