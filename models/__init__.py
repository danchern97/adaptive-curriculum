MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)
