import pickle

__all__ = [
    "save_pickle",
    "load_pickle",
]


def save_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)
    f.close()


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data
