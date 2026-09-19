from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

from molito.core._checks import PICKLE_PROTOCOL, check_type_all, check_unique

T = TypeVar("T")


class Vocabulary(Generic[T], Mapping):
    """Generic vocabulary class which maps tokens <--> indices."""

    def __init__(self, tokens: list[T]):
        check_unique(tokens, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map

    # *** Mapping Collection methods ***

    def __len__(self) -> int:
        return len(self.token_idx_map)

    def __getitem__(self, token: T) -> int:
        return self.get_index(token)

    def __contains__(self, token: T) -> bool:
        return token in self.token_idx_map

    def __iter__(self) -> Iterator[T]:
        return iter(self.token_idx_map)

    # *** Mapping functions ***

    def get_token(self, index: int) -> T:
        return self.idx_token_map[index]

    def get_index(self, token: T) -> int:
        return self.token_idx_map[token]

    def tokens_from_indices(self, indices: list[int]) -> list[T]:
        check_type_all(indices, int, "indices list")
        return [self.idx_token_map[idx] for idx in indices]

    def indices_from_tokens(self, tokens: list[T]) -> list[int]:
        return [self.token_idx_map[token] for token in tokens]

    # *** Check contents of vocab map ***

    def contains_token(self, token: T) -> bool:
        return token in self.token_idx_map

    def contains_index(self, index: int) -> bool:
        return index in self.idx_token_map

    # *** Iter functions ***

    def iter_tokens(self) -> Iterator[T]:
        return iter(self.token_idx_map)

    def iter_indices(self) -> Iterator[int]:
        return iter(self.idx_token_map)

    # *** Saving and loading functionality ***

    def to_bytes(self) -> bytes:
        tokens = list(self.token_idx_map.keys())
        return pickle.dumps(tokens, protocol=PICKLE_PROTOCOL)

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        return Vocabulary(pickle.loads(data))
