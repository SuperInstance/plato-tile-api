# plato-tile-api

Stateful tile API — process, search, score, rank in one interface.

Part of the [PLATO framework](https://github.com/SuperInstance) — deterministic AI knowledge management through tile-based architecture.

## Usage

```rust
use plato_tile_api::{Tile, TileAPI};

let mut api = TileAPI::new();
let results = api.process(vec![Tile {
    content: "Pythagorean triples snap to exact coordinates".into(),
    domain: "math".into(),
    confidence: 0.9,
}]);
let search = api.search("Pythagorean", 5);
let stats = api.stats();
```

Zero external dependencies.
