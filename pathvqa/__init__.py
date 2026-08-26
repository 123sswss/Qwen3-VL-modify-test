__all__ = [
    "PathVQAParquetStore",
    "PathVQADataset",
    "PathVQAStage1Dataset",
    "PathVQADataCollator",
]


def __getattr__(name):
    if name in __all__:
        from . import data_pipeline

        return getattr(data_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
