# docs

This folder is the companion site of the paper, for GitHub Pages. It shows, one record at a time, how the astronomy of a diary narrows its date. For each record of ADART I to III, it shows the tablet lines in translation and in cuneiform signs, the clauses that the parser reads and the language model checks and adds, the verdict of each clause in each of the 701 candidate years, the clocks and bits of each year, the strict set and the ranking. The page computes no verdict. It draws only the data that `scripts/site/build_site_data.py` writes from the committed results.

| File | What it holds |
|---|---|
| `index.html`, `site.css`, `site.js` | The page. `site.js` draws the charts with D3 7.9.0 from jsDelivr |
| `data/meta.json` | The candidate years, and one summary row per record for the record chooser |
| `data/records/<id>.json` | One record: its lines, clauses, verdicts, scores and sets |
| `data/sky.json` | The Babylonian longitude of the Sun and the five planets at the start and the end of each month of the 701 years |
| `img/sky/` | Drawings of the zodiac signs, and pictures of the Sun, the Moon and the planets |
| `img/scenes/` | Paintings for the sections of the page, the paper texture, the parchment and the picture for shared links |
| `img/museums/` | Drawings of the museum buildings |
| `img/periods/` | Emblems of the periods of rule, after ancient reliefs and coins |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are |

## Rebuild the data

Run the full rebuild of `scripts/README.md` first. Then run this from the repository root.

```bash
python scripts/site/build_site_data.py
```

It writes `docs/data/`. It does not change a number of the paper.

## Preview

```bash
python3 -m http.server 8765 --directory docs
```

Open http://localhost:8765 in a browser. The page reads `data/` through the server, so it does not open as a local file. It loads its fonts and D3 from the network.

## Credits

- Gemini 3 Pro Image made the drawings of the signs, the pictures of the planets, the paintings, the parchment, the drawings of the museums and the emblems. They are illustrations, not photographs or ancient objects.
- The rulers and the eras follow the tables of Parker and Dubberstein (1956), in `data/source/parker_dubberstein/`.
- The fonts are IM Fell English, IM Fell DW Pica, Roboto and Noto Sans Cuneiform, from Google Fonts.

## Licences

- The code is under the MIT License, in the root `LICENSE`.
- The data and figures are under CC BY 4.0.
- The translations of ADART I to III come from Astronomical Diaries Digital (ADsD), released under CC0.
- The cuneiform signs come from the ORACC Sign List (OSL), released under CC0.
