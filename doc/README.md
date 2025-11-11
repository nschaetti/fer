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

## Scientific Methodology
### Picbreeder Genome Training and Reconstruction
Picbreeder produces NEAT-style compositional pattern-producing networks (CPPNs). The repository replays those trained genomes so they can be compared with new optimizers. The process is deterministic and consists of four stages implemented in `src/process_pb.py`.

1. **Genome ingestion (`load_pbcppn`)**  
   Zipped XML genomes are parsed with `picbreeder_util.load_zip_xml_as_dict`. Every node is given a composite identifier `<branch>_<id>` so that distinct branches in the NEAT tree remain addressable once flattened. Node labels (`x`, `y`, `d`, `bias`, `hue`, `saturation`, `brightness`) are normalized so later stages can find coordinate inputs and HSV outputs reliably. When genomes expose the legacy `ink` channel, the loader clones it into three independent brightness/saturation/hue heads so the downstream CPPN always emits HSV.

2. **Forward replay (`do_forward_pass`)**  
   The genome is evaluated over a 256×256 Cartesian grid with coordinates  
   \[
   x,y \in [-1,1],\quad d = 1.4\sqrt{x^2 + y^2},\quad b = 1
   \]
   The evaluation recursively walks the NEAT graph from the HSV outputs toward the inputs. For each node \(n\), the incoming activations are aggregated linearly
   \[
   z_n = \sum_{(s,w)\in \text{In}(n)} w \cdot a_s
   \]
   and transformed with the activation in `activation_fn_map` (`sin`, `cos`, `tanh`, `gaussian`, etc.). Results are cached to avoid exponential recursion and cycle detection prevents infinite loops. The three terminal nodes are interpreted as HSV channels, wrapped into the valid ranges ((`h+1`) mod 1, `clip(s,0,1)`, `clip(|v|,0,1)`) and converted to linear RGB via `color.hsv2rgb`.

3. **Layerization and architecture extraction (`layerize_nn`)**  
   The NEAT DAG is topologically ordered by assigning a layer index to every node based on the deepest already-assigned parent. A `nodes_cache` tracks which nodes persist across layers; recurrent use of a node is represented by caching the previous activation instead of duplicating computation. From this cache, the algorithm derives a canonical architecture string  
   \[
   \texttt{<n_layers>;\<activation_1>:<width_1>,\ldots}
   \]
   where each width is the maximum concurrent usage of that activation anywhere in the network. This provides both the number of dense layers and the per-activation channel counts required to instantiate a Flax `CPPN`.

4. **Parameter transfer (`get_weight_matrix`, `FlattenCPPNParameters`)**  
   For every adjacent layer pair, `get_weight_matrix` constructs a dense weight matrix by locating the positional offset of each node inside the activation partitions described above. Cache edges become identity connections; novel nodes inherit their learned NEAT weights. The matrices are then inserted into the corresponding `Dense_i` kernels of a freshly initialized `CPPN`, and `FlattenCPPNParameters` converts that PyTree into a flat parameter vector that can be saved or optimized. The CLI (`python src/process_pb.py --zip_path ... --save_dir ...`) executes the entire pipeline and stores the rendered PNG, the native genome dictionary, the derived architecture string, and the flattened parameters so subsequent optimizers can treat the Picbreeder solution as an initialization point.

This replay procedure exactly mirrors the original Picbreeder training outcome because no further learning occurs—every operation is a deterministic transformation of the archived genome. The scientific value lies in being able to (a) inspect intermediate activations (`features` stored per layer), (b) measure architectural motifs in historical genomes, and (c) initialize gradient-based training from an evolved baseline.

### SGD-Based CPPN Training
`src/train_sgd.py` provides the gradient-based counterpart to Picbreeder’s evolutionary search. It optimizes the weights of a CPPN (defined in `src/cppn.py`) so that the generated image matches a user-specified RGB target.

1. **Model parameterization**  
   The CPPN accepts a structured coordinate tensor \((x,y,d,b)\) (and optional absolute axes) and pushes it through `n_layers` fully connected layers. Each layer concatenates parallel activation families whose widths are defined by the same architecture string used for Picbreeder replay. The final linear layer outputs raw HSV logits which `generate_image` maps to RGB using the same transformation as above. `FlattenCPPNParameters` exposes the parameters as a single vector, allowing optimizers that expect flat genomes to reuse the same representation.

2. **Loss function**  
   Given target image \(I^\*\) (loaded and normalized to `[0,1]`), and the current CPPN rendering \(I(\theta)\), the objective is the per-pixel mean-squared error  
   \[
   \mathcal{L}(\theta) = \frac{1}{3HW} \sum_{c,h,w} \left(I_{c,h,w}(\theta) - I^\*_{c,h,w}\right)^2.
   \]
   Computing the loss requires a full forward sweep over the 256×256 grid, which is automatically vectorized in JAX by `generate_image`.

3. **Optimization loop**  
   Parameters are initialized with either Lecun-style variance scaling (`init_scale="default"`) or a user-supplied scale factor. The script seeds all randomness via `--seed` to ensure reproducibility. Training creates an `optax.adam` optimizer with learning rate `--lr` and wraps it in a `TrainState`. One call to `train_step` performs:
   - Gradient evaluation with `jax.value_and_grad`.
   - Normalization of the gradient (\(g \leftarrow g / \|g\|_2\)) to decouple the step size from the loss scale, acting as an adaptive trust-region heuristic.
   - Parameter updates through Adam, keeping optimizer state on device.
   The main loop runs `n_iters` updates batched into blocks of 100 using `jax.lax.scan` for compilation efficiency. Progress bars expose the most recent loss, and the first 100 blocks also cache intermediate reconstructions for qualitative inspection.

4. **Outputs and experiment tracking**  
   After training, the script re-renders the final image, serializes the learned parameter vector, the architecture, the CLI arguments, and the loss history via the lightweight helpers in `src/util.py`. When `--save_dir` is supplied, the artifacts are placed under that directory (PNG for the final rendering plus pickle files for metadata). Because CPPNs are coordinate-based and not tied to a specific dataset, this workflow directly optimizes an arbitrary target without batching—each experiment is essentially full-image system identification.

With these pieces, the repository documents two complementary regimes: Picbreeder’s evolutionary replay for historically evolved CPPNs and a reproducible stochastic gradient descent procedure for training new CPPNs from scratch or fine-tuning the Picbreeder reconstructions.

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
