# SMILES vocabulary presets

Both JSON files contain complete, ordered vocabularies. A token's position in `tokens`
is its model index; special tokens are also listed in `special_tokens` to identify their
role. The files use the same versioned format as `StringVocab.to_bytes()`.

| File | Contents | Entries | Loader |
|---|---|---:|---|
| `smiles_char_vocab.json` | 74 characters and four special tokens | 78 | `StringVocab.characters()` |
| `smiles_regex_vocab.json` | Core chemical tokens, character fallback, and four special tokens | 169 | `StringVocab.smiles()` |

Loading does not filter, sort, or rebuild either preset. Keeping the final lists explicit
makes indices inspectable and stable even if the fitting policy changes. The full upstream
vocabulary is not bundled.

## Source and preparation

The regex preset's additional chemical tokens were selected from
[MolecularAI/MolBART's bart_vocab.txt](https://github.com/MolecularAI/MolBART/blob/fd2be528936d21386e51ce951f42442e82489c2e/bart_vocab.txt),
revision `fd2be528936d21386e51ce951f42442e82489c2e`. MolBART is also known as Chemformer.
The upstream repository's [Apache 2.0 license](https://github.com/MolecularAI/MolBART/blob/fd2be528936d21386e51ce951f42442e82489c2e/LICENSE)
is referenced here; its full text is not bundled. The source file contains 523 entries and
has SHA-256 `3b43e98c839813e2fefcef6b7401cae04019871bd2d78d52f1827252ea50c755`.

Molito modifications: select core chemical tokens, remove task/control labels and unused
slots, and exclude isotope/map enumerations and uncommon bracket forms. Original ordering
is retained. This is a new vocabulary, not compatible with pretrained Chemformer indices.

The one-time preparation used `StringVocab.is_core_token`:

- Single characters in the bounded SMILES alphabet, `Cl`, `Br`, and `%nn` ring tokens.
- Bracket atoms for B, C, N, O, P, S, Si and their applicable aromatic forms, optionally
  carrying `@`/`@@`, H/H1-H4, and charge +1/+2/-1/-2 (including shorthand +/-).
- `[H]`, `[H+]`, `[Li+]`, `[Na+]`, `[K+]`, `[Mg+2]`, `[Ca+2]`, `[Zn+2]`, and halide anions.

This retained 122 of 523 source entries. It is a vocabulary selection rule, not a
chemical validity test. Some retained forms require a suitable molecular context.
Unselected tokens remain representable via character fallback; their text is never changed.
The same predicate remains available for fitting new vocabularies to user data. It is not
applied when loading the shipped presets, and no filter string is needed in the JSON files.

To reproduce, download the pinned source file and apply:

```python
from pathlib import Path
from molito.core.vocab import StringVocab

source = Path("bart_vocab.txt").read_text().splitlines()
core_tokens = [token for token in source if StringVocab.is_core_token(token)]
vocab = StringVocab.characters().extend(core_tokens)
Path("smiles_regex_vocab.json").write_bytes(vocab.to_bytes())
```

During preparation, `<PAD>`, `<MASK>`, `<BOS>`, `<EOS>` and the sorted 74-character alphabet
were placed first, then retained tokens were appended without duplicates. Both files now
store those final indices explicitly. The alphabet consists of letters appearing in elements 1-118, aromatic/stereo
notation, digits, and `[]()@+-=#$:/\\.%*~<>`. There is no whitespace or arbitrary ASCII fallback.

The regex preset follows the [Molecular Transformer tokenizer](https://github.com/pschwllr/MolecularTransformer#pre-processing).
Molito adds vocabulary-aware character fallback and preserves regex-unmatched spans. Tests
cover every element symbol, targeted syntax examples, and exact encoding round trips on the
4,999 records in RDKit's bundled `Data/NCI/first_5K.smi` when that corpus is installed.
