from tqdm import tqdm


def track(obj, _track=True):
    if _track:
        return tqdm(obj)
    return obj
