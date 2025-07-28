DATASET_REGISTRY = {}


def register_dataset(name):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name, **kwargs):
    return DATASET_REGISTRY[name](**kwargs)
