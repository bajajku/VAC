from langgraph.graph import StateGraph
import os
class GraphImage:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    @staticmethod
    def create_graph_image(graph: StateGraph):
        graph_image = graph.get_graph().draw_mermaid_png()
        os.makedirs("images", exist_ok=True)
        with open("images/graph.png", "wb") as f:
            f.write(graph_image)
