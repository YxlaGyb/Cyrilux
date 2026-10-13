---
description: "the UTF-8 byte container's geometry and topology presets, declared as plain data."
kind: "package-reference"
---

# text

English | [中文](README.zh.md)

## Summary

`crates.modal.text` is the text modality's organ. It declares the container a byte stream lives in: the symbol domain, the padding position, the printable domain, the UTF-8 automaton that says which bytes start a character, which continue one, and how many continuations a lead asks for, plus the two runtime masks the network derives from those facts. It declares shapes only. There is no vocabulary and no merge rule, no per-symbol vector, no corpus frequency, no language or charset label, no scorer output, no learnable parameter and no tensor.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

The container.

| Export | Shape | Meaning |
|---|---|---|
| `ALPHABET` | `int` | Symbol domain cardinality, one symbol per byte value. |
| `PAD_ID` | `int` | The padding symbol written into the tail of a fixed-length window. |
| `PRINTABLE_START` | `int` | Lowest symbol of the printable domain; the bytes below it are control bytes. |

The UTF-8 automaton.

| Export | Shape | Meaning |
|---|---|---|
| `CONTINUATION` | `(int, int)` | Inclusive range of continuation bytes. |
| `OVERLONG_LEADS` | `(int, int)` | Inclusive range of overlong leads, which are illegal as character starts. |
| `LEAD_RANGES` | `tuple[(int, int)]` | The legal multi-byte lead ranges: 2, 3 and 4 byte sequences. |
| `LEAD_CONTINUATIONS` | `tuple[int]` | Continuation count parallel to `LEAD_RANGES`. |
| `START_RANGES` | `tuple[(int, int)]` | Every legal start: ASCII plus the three lead ranges. |
| `CONTINUATION_PREFIX` | `(int, int)` | The bit-prefix test `(b & first) != second`, the one predicate a tensor side can apply without a gather. |
| `is_continuation(b)` | `bool` | Python predicate over the continuation range. |
| `is_start(b)` | `bool` | Python predicate over `START_RANGES`. |
| `lead_length(b)` | `int` | Continuation count a lead asks for; 0 for ASCII and for illegal leads. |

Derived tables, one entry per symbol.

| Export | Shape | Meaning |
|---|---|---|
| `UTF8_BLOCK_CNT` | `tuple[bool]` | Position may not continue a character: a lead or an ASCII byte. |
| `UTF8_BLOCK_START` | `tuple[bool]` | Position may not start a character: a continuation byte or an overlong lead. |
| `PRINT_MASK` | `tuple[bool]` | The printable domain, as a mask the network materialises at construction. |

```python
from crates.modal.text.geometry import ALPHABET, LEAD_RANGES, CONTINUATION_PREFIX
```

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

`geometry.py` is the whole package, and it imports nothing but `__future__`. It holds plain `int`, `tuple` and `bool` values, so three kinds of reader share one copy of the facts: the network materialises its runtime buffers from them, a tool reads the padding position from them while doing its own torch work, and plain Python reads them as data.

The organ declares shapes; it never builds tensors. That boundary is not a blanket ban on `torch` in the repository (`crates.utils.codec` imports it): a preset that held a tensor would also own its dtype, its device, its lifetime and its `state_dict` key, and those are `crates.model`'s construction policy and the schema that `HybridPCNet.load` fails fast on. One fact has one owner, so the masks are built where the buffers are built. It also marks the line between declaring and processing: a modality whose geometry needs computation would use `math`, and a modality that needs `torch` is transforming a signal, which makes it a tool.

The two generation masks are the byte-domain facts the forward path actually consumes: `UTF8_BLOCK_CNT` and `UTF8_BLOCK_START` become the two bool buffers `crates.model` registers for the UTF-8 boundary checks, and `PRINT_MASK` becomes the fp16 symbol mask the logits carry, so a byte outside the print domain is never chosen. The network reads these presets, never the other way round.

Two report rulers deliberately stay outside this package: `scripts/world_lang.py` and `scripts/audit_char_coverage.py` carry their own byte classification, because a ruler that measures the model's language output must not take the model's own domain as its ruler.

-----

<a id="further-exploration"></a>
## Further exploration

- [modal](../README.md): the package group that owns the body and tool boundary.
- [model](../../model/README.md): the network that materialises these presets into runtime masks.
- [utils](../../utils/README.md): the codec and corpus tools that read these presets.
