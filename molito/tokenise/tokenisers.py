from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from molito.core.vocab.string import StringVocab

# Molecular Transformer (Schwaller et al., 2019). Unmatched spans are retained by
# our scanner, including invalid strings and newer SMILES extensions.
SMILES_REGEX = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"


class Tokeniser(ABC):
    """Lossless text splitting and vocabulary encoding, independent of chemistry."""

    def __init__(self, vocab: StringVocab):
        self.vocab = vocab

    @abstractmethod
    def tokenise(self, text: str) -> list[str]:
        """Split text without discarding or normalising any characters."""

    @staticmethod
    def detokenise(tokens: Iterable[str]) -> str:
        return "".join(tokens)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text, optionally surrounding it with <BOS> and <EOS>.

        Unsupported characters raise rather than being replaced by an unknown token.
        Extend the vocabulary explicitly to support a larger input alphabet.
        """

        tokens = self.tokenise(text)
        if add_special_tokens:
            if not {"<BOS>", "<EOS>"}.issubset(self.vocab.special_tokens):
                raise ValueError("Boundary encoding requires <BOS> and <EOS> special tokens.")
            tokens = ["<BOS>", *tokens, "<EOS>"]

        missing = sorted(set(tokens).difference(self.vocab))
        if missing:
            raise ValueError(
                f"Tokens {missing!r} are outside the vocabulary; extend it explicitly to encode this text."
            )
        return self.vocab.indices_from_tokens(tokens)

    def decode(self, indices: Iterable[int], skip_special_tokens: bool = False) -> str:
        """Decode indices; removing control tokens is explicit and potentially lossy."""

        tokens = [self.vocab.get_token(idx) for idx in indices]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.vocab.special_tokens]
        return self.detokenise(tokens)

    def to_bytes(self) -> bytes:
        obj = {"version": 1, "kind": "char", "vocab": json.loads(self.vocab.to_bytes())}
        if isinstance(self, RegexTokeniser):
            obj.update(kind="regex", pattern=self.pattern, flags=self.flags, extra_tokens=self.extra_tokens)
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> Tokeniser:
        obj = json.loads(data)
        if obj["version"] != 1:
            raise ValueError("Unsupported tokeniser version.")
        vocab = StringVocab.from_bytes(json.dumps(obj["vocab"]).encode("utf-8"))
        if obj["kind"] == "char":
            return CharTokeniser(vocab)
        if obj["kind"] == "regex":
            return RegexTokeniser(obj["pattern"], vocab, flags=obj["flags"], extra_tokens=obj["extra_tokens"])
        raise ValueError(f"Unknown tokeniser kind {obj['kind']!r}.")

    def save(self, path: str | Path) -> None:
        with Path(path).open("xb") as stream:
            stream.write(self.to_bytes())

    @staticmethod
    def load(path: str | Path) -> Tokeniser:
        return Tokeniser.from_bytes(Path(path).read_bytes())


class CharTokeniser(Tokeniser):
    """One token per character, with a fixed SMILES alphabet by default."""

    def __init__(self, vocab: StringVocab | None = None):
        super().__init__(StringVocab.characters() if vocab is None else vocab)
        if any(len(token) != 1 for token in self.vocab if token not in self.vocab.special_tokens):
            raise ValueError("Character vocabularies can only contain characters and special tokens.")

    def tokenise(self, text: str) -> list[str]:
        return list(text)


class RegexTokeniser(Tokeniser):
    """Regex matching with vocabulary-aware, lossless character fallback.

    Args:
        pattern: Matching expression. Capturing groups do not change token extraction.
        vocab: Vocabulary; defaults to characters only. Use `smiles` for the bundled preset.
        flags: Python regular expression flags.
        extra_tokens: Literal tokens to match before the regex, longest first. These are
            added to the vocabulary without renumbering existing entries. Reserved special
            tokens cannot be matched in ordinary text.
    """

    def __init__(
        self,
        pattern: str,
        vocab: StringVocab | None = None,
        flags: int = 0,
        extra_tokens: Iterable[str] = (),
    ):
        vocab = StringVocab.build() if vocab is None else vocab
        extras = tuple(dict.fromkeys(extra_tokens))
        if any(not token or token in vocab.special_tokens for token in extras):
            raise ValueError("Literal extra tokens must be nonempty and cannot be special tokens.")
        super().__init__(vocab.extend(extras))
        self.pattern = pattern
        self.flags = int(flags)
        self.extra_tokens = extras

        literals = [re.escape(token) for token in sorted(extras, key=lambda token: (-len(token), token))]
        self._regex = re.compile(pattern, flags)
        self._literal_regex = re.compile("|".join(literals), flags) if literals else None
        if self._regex.match("") is not None:
            raise ValueError("Token patterns must not match empty strings.")

    @staticmethod
    def smiles(extra_tokens: Iterable[str] = ()) -> RegexTokeniser:
        return RegexTokeniser(SMILES_REGEX, StringVocab.smiles(), extra_tokens=extra_tokens)

    def split(self, text: str) -> list[str]:
        """Regex candidates before vocabulary fallback, retaining every unmatched span."""

        tokens = []
        offset = 0
        while offset <= len(text):
            regex_match = self._regex.search(text, offset)
            literal_match = self._literal_regex.search(text, offset) if self._literal_regex is not None else None
            matches = [match for match in (literal_match, regex_match) if match is not None]
            if not matches:
                break
            match = min(matches, key=lambda match: match.start())
            if match.start() == match.end():
                raise ValueError("Token patterns must not produce zero-width matches.")
            tokens.extend(text[offset : match.start()])
            tokens.append(match.group(0))
            offset = match.end()
        tokens.extend(text[offset:])
        return tokens

    def tokenise(self, text: str) -> list[str]:
        tokens = []
        for token in self.split(text):
            if token in self.vocab and token not in self.vocab.special_tokens:
                tokens.append(token)
            else:
                tokens.extend(token)
        return tokens

    def fit(
        self,
        texts: Iterable[str],
        min_frequency: int = 1,
        max_tokens: int | None = None,
        token_filter: Callable[[str], bool] | None = StringVocab.is_core_token,
        extra_tokens: Iterable[str] = (),
    ) -> RegexTokeniser:
        """Return a fitted tokeniser; the original vocabulary is never modified.

        Existing entries are retained. `max_tokens` limits newly learned tokens, not
        mandatory characters or explicit extras. Ties are sorted lexically. Pass
        `token_filter=None` to learn all regex candidates, including mapped atoms.
        """

        if min_frequency < 1 or (max_tokens is not None and max_tokens < 0):
            raise ValueError("min_frequency must be positive and max_tokens must be nonnegative.")
        counts = Counter(token for text in texts for token in self.split(text))
        candidates = [
            token
            for token, count in counts.items()
            if count >= min_frequency and token not in self.vocab and (token_filter is None or token_filter(token))
        ]
        candidates.sort(key=lambda token: (-counts[token], token))
        learned = candidates if max_tokens is None else candidates[:max_tokens]
        vocab = self.vocab.extend(learned)
        return RegexTokeniser(self.pattern, vocab, self.flags, [*self.extra_tokens, *extra_tokens])
