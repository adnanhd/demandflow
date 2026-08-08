from typing import Any


class MemoStore:
    """In-memory content-addressed cache."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> tuple[bool, Any]:
        if key in self._store:
            return True, self._store[key]
        return False, None

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    def has(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def keys(self) -> list[str]:
        return list(self._store)
