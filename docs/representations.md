# Molecular representations and tokenisation

`MolRepr` is the small-molecule interface shared by `GraphMol`, `RDKitMol`, and `StringMol`.
`SmilesMol` implements `StringMol`. Proteins and complexes retain their existing interfaces.

## Exact strings and conversion

```python
from molito import GraphMol, RDKitMol, SmilesMol

mol = SmilesMol("OCC", meta={"id": "ethanol"})
mol.text                                  # "OCC": original spelling retained
graph = mol.to(GraphMol)                   # compact training representation
wrapped = graph.to(RDKitMol)               # owned RDKit object
canonical = wrapped.to(SmilesMol)          # .text is "CCO"
text = graph.to_smiles()                   # a plain Python string
raw = wrapped.to_rdkit()                   # independent RDKit copy
```

Every `.to(...)` conversion copies metadata. Same-representation conversion with no extra
options returns a copy without parsing. Constructors for strings accept invalid text:

```python
invalid = SmilesMol("C1[")
invalid.save("generated.molito")           # exact text and metadata, versioned JSON
restored = SmilesMol.load("generated.molito")
# invalid.validate()                      # raises ConversionError
# invalid.to(GraphMol)                    # raises ConversionError
```

`validate()` checks RDKit sanitisation, allowing disconnected molecules such as salts and mixtures.
Use `validate(connected=True)` to also require exactly one connected component. This rejects
empty molecules as well as multiple fragments; a single atom passes. The check uses molecular
bonds, not the presence of `.` in the text, and never removes fragments or changes the input.
Failed checks raise `ConversionError` for all three small-molecule representations.

Successful tokenisation does not imply valid chemistry. SMILES parsing retains explicit hydrogen atoms
and rejects trailing molecule names and CXSMILES annotations; these are not plain SMILES.
`to_rdkit(sanitise=False)` can produce unsanitised objects and returns `None` on parsing failure.
Errors reading lazy data, including closed files and corrupt payloads, propagate to the caller.
`.to(...)` requests sanitisation. The original input is not mutated.

## Preservation limits

`RDKitMol` owns a copy of its input. Its `rdkit_mol` property explicitly exposes the mutable
owned object; `copy()` and `to_rdkit()` return independent objects. Binary persistence stores
molecule, atom, bond, and conformer properties, conformers with double-precision coordinates,
and the writing RDKit version. It uses RDKit's native molecular binary format, not Python
object pickle. Compatibility with a particular older RDKit reader is not guaranteed.

SMILES does not carry conformers or arbitrary properties. `GraphMol` stores the supported
atomic/bond features, tetrahedral/E/Z stereo, and optional float32 conformer arrays, but does
not store atom maps, isotope labels, or arbitrary RDKit properties. Its supported bond and
stereo types are also narrower than RDKit's. Unsupported bond types raise a conversion error.

Use `mol.to(TargetClass, strict=True)` to reject detected loss. Checks cover molecular
structure and CX annotations, atom order/state, noncomputed RDKit properties, and conformers.
Coordinate comparisons allow float32 rounding. Strict mode is conservative: canonical atom
reordering or changes to explicit/implicit hydrogen bookkeeping can also be rejected. It
does not guarantee arbitrary RDKit extensions or attached Python attributes. Exact original
SMILES spelling is only retained by copying or storing the original `SmilesMol`.

## Tokenisers

```python
from molito.core import StringVocab
from molito.tokenise import CharTokeniser, RegexTokeniser

chars = CharTokeniser()
tokens = RegexTokeniser.smiles()

mol = SmilesMol("[13CH3:147][C@@H](Cl)Br")
pieces = mol.tokenise(tokens)
# ['[', '1', '3', 'C', 'H', '3', ':', '1', '4', '7', ']', '[C@@H]', '(', 'Cl', ')', 'Br']
ids = mol.encode(tokens)
assert tokens.decode(ids) == mol.text
assert "".join(pieces) == mol.text
```

Both default vocabularies use a bounded **74-character alphabet**: letters used by the
118 element symbols and stereo/aromatic notation, digits, and SMILES punctuation. It includes
`.`, `*`, `@`, `/`, `\`, `%`, brackets, charges, and bond notation. Spaces and arbitrary Unicode
are not included. Tokenisation still preserves them; encoding reports missing vocabulary
entries. Add characters explicitly if your input needs them.

The regex preset uses Molecular Transformer matching and a filtered MolBART/Chemformer
vocabulary. It retains **122 of 523 upstream entries** and adds character fallback and model
tokens, giving **169 vocabulary entries**. Mapped atoms, isotopes, and unusual bracket atoms
fall back to characters if absent; they are never stripped from the input. This vocabulary
does not preserve pretrained Chemformer model indices. The complete ordered vocabularies
are stored in `molito/defs/smiles_char_vocab.json` (78 entries including specials) and
`molito/defs/smiles_regex_vocab.json` (169 entries). `StringVocab.characters()` and
`StringVocab.smiles()` load them directly, without filtering or rebuilding their indices.
See `molito/defs/smiles_vocab.md` for the pinned source, upstream license link, and preparation rules.

Token count depends on the tokeniser. String molecules do not expose a global `tokens` or
`seq_length` property, and string vocabularies do not use `VocabConfig`.

### Model tokens and extensions

`<PAD>` has index 0; `<MASK>`, `<BOS>`, and `<EOS>` are reserved model-control tokens.
Boundary insertion and removal are explicit:

```python
ids = tokens.encode("C.O", add_special_tokens=True)
assert tokens.decode(ids) == "<BOS>C.O<EOS>"
assert tokens.decode(ids, skip_special_tokens=True) == "C.O"

custom = RegexTokeniser.smiles(extra_tokens=["[13CH3]", "<CUSTOM>"])
assert custom.tokenise("[13CH3]<CUSTOM>") == ["[13CH3]", "<CUSTOM>"]

# For character vocabularies, extend with individual characters.
extended_chars = CharTokeniser(chars.vocab.extend([" ", "é"]))
```

Literal custom tokens match before the regex, longest first. A vocabulary extension preserves
existing indices and returns a new object. A literal spelling such as `<MASK>` in input text
is split as ordinary text, not treated as a model-control token. Add masking/padding IDs
explicitly in your training pipeline. The library does not introduce an unknown token or
silently grow the vocabulary during encoding. Existing graph `PAD`/`MASK` tokens are unchanged.

### Custom regexes and fitting

```python
from molito.tokenise import SMILES_REGEX, Tokeniser

regex = RegexTokeniser(r"(Cl)|(Br)", StringVocab.build(["Cl", "Br"]))
assert regex.tokenise("ClCBr") == ["Cl", "C", "Br"]

# split() gives direct regex candidates without vocabulary fallback.
assert regex.split("ClC!Br") == ["Cl", "C", "!", "Br"]

base = RegexTokeniser(SMILES_REGEX)           # character vocabulary initially
fitted = base.fit(training_smiles, min_frequency=5, max_tokens=256)
fitted.save("tokeniser.json")
restored = Tokeniser.load("tokeniser.json")
```

Fitting retains existing entries and adds candidates ordered by descending frequency with
lexical tie-breaking. `max_tokens` limits newly learned entries, not mandatory/existing ones.
The default core-token filter excludes mapped and isotope-labelled bracket atoms. Pass
`token_filter=None` to learn every candidate, or a predicate for a custom selection policy;
`extra_tokens` bypass filtering. Fit on the training split and reuse the saved artifact for
validation and inference. JSON persistence includes regex flags, ordered vocabulary, literal
extras, special tokens, and a format version; the fitting predicate is not needed at inference.

Custom regexes preserve capturing-group independence and unmatched spans. Zero-width matches
raise an error. Matching is lexical, not a SMILES validity check.

## Batches and storage

```python
from molito import MolBatch

batch = MolBatch([SmilesMol("OCC"), SmilesMol("C1[")])
batch.save("smiles_dataset/", shard_size=1000)
with MolBatch.load("smiles_dataset/") as loaded:
    assert loaded[1].text == "C1["

valid = MolBatch([SmilesMol("CCO"), SmilesMol("CCN")])
graphs = valid.to(GraphMol)                  # GraphBatch, with graph array operations
strings = graphs.to(SmilesMol)               # MolBatch
```

Native batches are homogeneous and support indexing, subsets, metadata columns, and sharded
HDF5 persistence. String shards store exact UTF-8 bytes; RDKit shards store native binary
payloads. Both reuse molito's JSON/columnar metadata storage. An empty batch needs
`MolBatch([], mol_type=SmilesMol)`. Conversion failures report the record index and never
silently drop records.

Native batch loading leaves molecule payloads on disk in both modes:

| Option | Molecule wrappers | Text / RDKit payloads |
| --- | --- | --- |
| `materialise=True` (default) | Created at load time | Read when accessed |
| `materialise=False` | Created on each lookup | Read when accessed |

Offsets and JSON metadata are read up front. Columnar metadata values remain on disk until
accessed, and `meta_column()` on a deferred batch reads metadata without creating molecule
wrappers. RDKit binaries decode as whole molecules; each wrapper caches its decoded RDKit
object so mutations through `.rdkit_mol` persist on that wrapper. String text is read on access.

With `materialise=False`, repeated `batch[i]` calls return fresh wrappers. Hold a wrapper or
use `subset()` when making in-memory edits. Loaded metadata is read-only, as with graph batches;
use `.read()` for an independent mutable copy or assign a new metadata dictionary locally.

Loaded batches own open HDF5 files. Subsets borrow their source files, so closing a subset
leaves its source open. Call `.read()` to keep data after the owning batch closes:

```python
with MolBatch.load("smiles_dataset/", materialise=False) as loaded:
    train = loaded.subset([0, 1]).read()

assert train[1].text == "C1["                # independent of the closed file
```

`StringMol` and `RDKitMol` also provide `.read()` for a single molecule. Edits affect the
in-memory object; write a new dataset with `.save()` to persist them. Single-molecule `.load()`
still reads its native byte file eagerly. Existing 0.2.0 batch files need no migration.

Custom string subclasses are selected using `mol_type=YourStringMol`. Lazy construction uses
the `_from_lazy(payload, meta)` classmethod without calling `__init__`; subclasses with extra
state should override that factory and initialise the extra state without reading the payload.

Existing `GraphBatch` storage, lazy loading, vocabulary imports, and graph indices are
unchanged. Native shards have their own representation/version markers and must be opened
with `MolBatch`. Existing GraphMol/GraphBatch `to_bytes`/`from_bytes` use Python pickle;
only load those graph byte files from trusted sources.
