import time

import chromadb


def create_persistent_client(path, retries=3, delay=0.25):
    last_error = None

    for _attempt in range(retries):
        try:
            return chromadb.PersistentClient(path=str(path))
        except Exception as exc:
            last_error = exc
            time.sleep(delay)

    raise last_error
