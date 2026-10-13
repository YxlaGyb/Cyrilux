---
description: "the package group of the project's modalities, holding each signal's geometry and topology presets."
kind: "package-group"
---

# modal

English | [中文](README.zh.md)

## Summary

`crates.modal` is the package group of the project's modalities. Each subpackage is one signal's organ: it declares the geometry and topology of that signal's container, that is the symbol domain, the padding position, and which byte positions may start or continue a character. A modality is a leaf: it imports no network code, so adding one adds a subpackage without moving the network's topology. Everything that processes a corpus stays outside this group: reading samples, cutting windows, converting and splitting datasets are tools in `crates.utils`, driven from `crates.cli`.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further exploration](#further-exploration)

-----

<a id="use-this-package"></a>
## Use this package

| Package | Signal | Organ |
|---|---|---|
| [text](text/README.md) | UTF-8 byte stream | Byte container geometry: symbol domain, padding position, printable domain, UTF-8 byte classification, and the generation masks derived from it. |

```python
from crates.modal.text.geometry import ALPHABET, PRINTABLE_START
```

The byte codec and the window segmentation that move a corpus into this container are processing tools, not organs: they live in [utils](../utils/README.md) and obey these presets.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

The boundary this group draws is the one between how a signal is shaped and how a corpus is processed. Geometry and topology presets describe the container a signal lives in, so they belong to the modality. Reading samples, cutting windows, converting corpora and splitting corpora all transform data, so they belong to `crates.utils`, driven from `crates.cli`.

Layering runs one way, and one edge is deliberately allowed: the network imports a modality's presets (`crates.model` builds its runtime masks from `crates.modal.text.geometry`, and `crates.hybrid` reads the byte-domain constants its forward, readout and learning paths need), and tools import them too, so `crates.utils.codec` takes the padding position from the organ instead of restating it. A modality imports neither the network nor the tools, so it is a leaf beside `crates.kernels`.

Sharing is deliberately absent inside this group. A preset that both text and a future modality read would force one signal's shape onto the other, so every subpackage declares its own; adding a modality means adding one sibling subpackage and changing nothing else in the repository.

-----

<a id="further-exploration"></a>
## Further exploration

- [text](text/README.md): the text modality's byte-domain presets.
- [model](../model/README.md): the network that builds its runtime masks from these presets.
- [utils](../utils/README.md): the corpus and codec tools that deliberately live outside this group.