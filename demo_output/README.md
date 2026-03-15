Demo output and explanation for the headless MDS example

Files produced
- `lucid-viz/demo_output/matrix.png` — heatmap of the similarity matrix (values in [0,1]).
- `lucid-viz/demo_output/mds.png` — classical MDS scatter; edges shown for similarity ≥ 0.6.
- `lucid-viz/demo_output/mds_annot.png` — same scatter with all labels and node sizes by weighted degree.
- `lucid-viz/demo_output/coords.csv` — CSV with computed 2D coordinates (label,x,y).

What this demo does
- Reads the example adjacency CSV at `tests/fixtures/sample_adjacency.csv`.
- Converts similarity → distance with `d = 1 - s` and runs classical MDS to produce 2D coordinates.
- Produces a matrix heatmap and two visualizations of the MDS layout (thresholded and annotated).

How to reproduce locally
1. From repository root, create a Python venv and install dependencies (one-liner):
   ```bash
   python3 -m venv demo/python/venv && demo/python/venv/bin/pip install --upgrade pip setuptools wheel && demo/python/venv/bin/pip install numpy matplotlib
   ```
2. Run the demo script:
   ```bash
   demo/python/venv/bin/python demo/python/demo.py
   ```
3. Output images and `coords.csv` will appear in `lucid-viz/demo_output`.

Quick interpretation guide
- `matrix.png`: bright cells ≈ high similarity (close to 1.0). Diagonal is 1.0 (self-similarity).
- `mds.png`: nodes are placed to reflect pairwise distances (derived from 1 − similarity). Edges are drawn for similarities ≥ 0.6 so you can see strong connections.
- `mds_annot.png`: same layout but with every node labeled and node sizes proportional to weighted degree (sum of similarities to other nodes).

Example (from generated `coords.csv`)
- `word1: (-0.217402, -0.159798)` — coordinates are in an arbitrary scale; spatial proximity encodes similarity.

Interpretation tips
- Strong similarity (e.g., 0.82) → small distance (0.18) → nodes placed close together.
- Thresholding edges (as in `mds.png`) reduces visual clutter and highlights strong structure.

Next suggestions
1. Swap the conversion method to `sqrt(1 - s^2)` or `-ln(s)` and compare layouts.
2. Run SMACOF for iterative stress minimisation starting from these coordinates for a potentially better fit.
3. Use `coords.csv` as input to the GUI (`lv-app`) or to an export script to render high-quality images.

If you want, I can now create a short annotated markdown that embeds the three images inline (ready for viewing), or I can run a variant that uses SMACOF and adds stress values to the output. Which would you prefer?
