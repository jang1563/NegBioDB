"""ML baseline models for DTI binary prediction.

Requires torch (install with `pip install negbiodb[ml]`).
"""

__all__ = ["DeepDTA", "GraphDTA", "DrugBAN"]


def __getattr__(name: str):
    if name in ("DeepDTA", "GraphDTA", "DrugBAN"):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"negbiodb.models.{name} requires torch. "
                "Install with: pip install negbiodb[ml]"
            ) from e
        if name == "DeepDTA":
            from negbiodb.models.deepdta import DeepDTA
            return DeepDTA
        if name == "GraphDTA":
            from negbiodb.models.graphdta import GraphDTA
            return GraphDTA
        if name == "DrugBAN":
            from negbiodb.models.drugban import DrugBAN
            return DrugBAN
    raise AttributeError(f"module 'negbiodb.models' has no attribute {name!r}")
