from typing import Any, Callable, Optional, Type
from langgraph.graph import StateGraph, START, END
from models.state import State

'''
Place holder for the custom agent.
'''

def Agent(
    *,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
    impl: list[tuple[str, Callable]],
) -> StateGraph:
    """Create the state graph for CustomAgent."""
    # Declare the state graph
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "model",
        "tools",
        "route_after_model",
    }

    missing_nodes = expected_implementations - all_names
    if missing_nodes:
        raise ValueError(f"Missing implementations for: {missing_nodes}")

    extra_nodes = all_names - expected_implementations

    if extra_nodes:
        raise ValueError(
            f"Extra implementations for: {extra_nodes}. Please regenerate the stub."
        )

    # Add nodes
    builder.add_node("model", nodes_by_name["model"])
    builder.add_node("tools", nodes_by_name["tools"])

    # Add edges
    builder.add_edge(START, "model")
    builder.add_edge("tools", "model")
    builder.add_conditional_edges(
        "model",
        nodes_by_name["route_after_model"],
        [
            "tools",
            END,
        ],
    )
    return builder