# AMReX-Tecplot-FE-Export
A Tecplot FE exporter for AMReX, supporting direct MultiFab output and TecIO-based binary writing.

## Overview

This repository provides a Tecplot FE exporter for AMReX-based simulations.  
It supports direct output from existing `MultiFab` data without first reading a plotfile, and is designed for a robust workflow in MPI runs:

1. gather distributed `MultiFab` data to rank 0,
2. construct the FE mesh on rank 0,
3. write Tecplot-readable output in ASCII or binary format.

This approach is more stable than assembling the FE mesh directly across MPI ranks, especially when output is only performed occasionally during a simulation.

## Features

- Direct export from existing AMReX `MultiFab` data
- Tecplot FE output
  - ASCII (`.dat`)
  - optional TecIO binary (`.dat`)
- Rank-0 gather-and-write workflow for MPI runs
- Works with cell-centered AMReX data
- Supports FE mesh construction from AMR grid data
- Suitable for integration into existing `AmrCoreAdv`-style applications

## Why rank-0 gather and serial output?

In practice, writing Tecplot FE files directly in parallel is error-prone because it involves:

- node numbering consistency across ranks
- element connectivity offsets
- processor-boundary duplication
- data layout consistency during MPI collation

Since Tecplot output is usually performed only every few steps, a simpler and more robust strategy is:

- keep the simulation fully parallel,
- gather the output data to rank 0,
- perform FE construction and file writing on rank 0 only.

This repository follows that strategy.

## Requirements

- Linux
- C++ compiler with AMReX support
- AMReX
- MPI
- Optional: Tecplot TecIO library for binary `.plt` output

## Optional TecIO support

Binary Tecplot output requires TecIO.

If TecIO is enabled, make sure your build can find:

- `TECIO.h`
- `libtecio.so`

and that the runtime loader can also find the shared library, e.g. via:

```bash
export LD_LIBRARY_PATH=/path/to/Tecplot/bin:$LD_LIBRARY_PATH
