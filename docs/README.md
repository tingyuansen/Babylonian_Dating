# Interactive page

This folder holds the interactive page at https://tingyuansen.github.io/Babylonian_Dating/. The page follows one record at a time from its text to its candidate years, with its clauses, the verdict of each clause in each candidate year, and the decision. Its last chapter loads the game A Little Babylon (https://tingyuansen.github.io/Babylonian_Observatory/), which shows the agreeing statements of the chosen record in the sky over Babylon; the page and the game keep the same record. GitHub Pages publishes this folder.

| File | What it holds |
|---|---|
| `index.html` | The page |
| `site.js` | The code that draws the page, with D3 |
| `site.css` | The style of the page |
| `data/` | The data of the page, from `make site` (`scripts/site/build_site_data.py`). `meta.json` holds the candidate years and one summary row per record. `sky.json` holds the longitude of the Sun and the five planets at the start and the end of each month of the 701 years. `records/` holds one file for each of the 417 dated records of ADART I to III. |
| `img/` | The images of the page. The illustrations are made by Gemini 3 Pro Image and show no real object; `scenes/ch4.webp` is a view from the game. |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are |

To see the page on your computer, run `python3 -m http.server 8765 --directory docs` and open http://localhost:8765. The page loads D3 and its fonts (IM Fell English, IM Fell DW Pica, Roboto and Noto Sans Cuneiform) from the network.
