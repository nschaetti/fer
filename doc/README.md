# FER Research Repository Documentation

## Overview
This repository contains the code used to process Picbreeder genomes, convert them into compositional pattern-producing networks (CPPNs), and optimize those CPPNs to recreate target images. The `src/` tree holds the actively maintained scripts while `assets/`, `data/`, and `picbreeder_genomes/` retain reference artifacts from prior experiments.

## Environment Setup
1. Create and activate a Python environment (Python 3.10+ is recommended).
2. Install dependencies: `pip install -r requirements.txt`.
3. (Optional) Install Jupyter for exploring the notebooks stored under `src/`.

## Module Reference
### `src/color.py`
Contains `hsv2rgb`, a JAX-compatible HSV to RGB converter used by every renderer. Inputs and outputs are normalized to `[0, 1]` and the function is vectorized, so it can operate on entire image grids.

### `src/cppn.py`
Defines the Flax-based `CPPN` module along with a `FlattenCPPNParameters` utility that reshapes Flax parameter PyTrees into flat vectors (and back) using EvoSax. `generate_image` evaluates the CPPN over a Cartesian grid composed of normalized coordinates, radial distance, bias, and optional absolute axes to produce RGB images.

### `src/picbreeder_util.py`
Implements XML parsing helpers that load zipped Picbreeder genomes and convert them into dictionary structures. These utilities normalize node/link attributes and guarantee that the resulting mapping always exposes a `genome` key.

### `src/process_pb.py`
Provides the tooling that ingests a Picbreeder genome (`load_pbcppn`), performs a forward pass over the discovered NEAT graph (`do_forward_pass`), and converts that sparse representation into a dense CPPN architecture (`layerize_nn`). The module also exposes a CLI (`python src/process_pb.py --zip_path <path> --save_dir <dir>`) that saves the rendered image, serialized genome, architecture string, and flattened parameters.

### `src/train_sgd.py`
Implements a command-line workflow for fitting CPPNs to target images via Adam/SGD. Supply an architecture string (`--arch`) and the path to a reference PNG (`--img_file`). Intermediate reconstructions and the loss curve are tracked so they can be inspected later. Example: `python src/train_sgd.py --arch "12;cache:15,gaussian:4" --img_file data/sgd_apple/img.png --save_dir runs/apple_fit`.

### `src/util.py`
Houses lightweight JSON and pickle persistence helpers that every CLI uses when writing results.

## Command-Line Workflow Details
### Genome Layerization (`process_pb.py`)
- `--zip_path`: Path to a Picbreeder genome ZIP (mandatory).
- `--save_dir`: Optional directory used to dump the rendered PNG, NEAT graph, CPPN architecture, and flattened parameters.
- Internally the script layerizes the NEAT topology to match the Flax CPPN layout so the saved parameters can be reloaded with `FlattenCPPNParameters`.

### SGD Training (`train_sgd.py`)
- `--arch`: Architecture specification such as `"6;sin:8,gaussian:8,identity:4"`.
- `--img_file`: Target RGB image trimmed to three channels.
- `--seed`, `--lr`, `--n_iters`, and `--init_scale`: Usual optimizer knobs.
- Outputs include the trained parameters, the produced PNG, and serialized metadata (arguments, architecture, losses).

## Data Artifacts
- `data/` contains sample optimization outputs grouped by motif (apple, butterfly, skull).
- `picbreeder_genomes/` stores zipped genomes that can be fed into `process_pb.py`.
- `assets/` holds figures used in publications (not required for running the code but useful for context).

## Development Notes
- Every script in `src/` (and the mirrored `src/fer/src/` subtree) now ships with Google-style docstrings to aid comprehension.
- End-of-block comments (`# end if`, `# end for`, etc.) were added throughout to make nesting depth explicit during research handoffs.
- When adding new scripts, follow the same documentation and commenting conventions so the research log remains searchable and self-explanatory.
