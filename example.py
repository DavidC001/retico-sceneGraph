from retico_vision import ScreenModule, WebcamModule
from retico_core.debug import DebugModule
from retico_core import network
from retico_sceneGraph import SceneGraphModule, SceneGraphDrawingModule, SceneGraphMemory

webcam = WebcamModule(meta_data={"camera_name": "office"})
screen = ScreenModule()
scene_graph = SceneGraphModule()
scene_graph_drawing = SceneGraphDrawingModule()
scene_graph_memory = SceneGraphMemory()
debug = DebugModule()


webcam.subscribe(scene_graph)
# webcam.subscribe(screen)
# scene_graph.subscribe(debug)
scene_graph.subscribe(scene_graph_drawing)
scene_graph_drawing.subscribe(screen)
scene_graph.subscribe(scene_graph_memory)

network.run(webcam)

# input("Press Enter to stop the network...")
breakpoint()
print("Stopping the network...")

network.stop(webcam)