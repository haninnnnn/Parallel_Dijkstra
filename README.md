
# Parallel Dijkstra's Algorithm (OpenMP + MPI)

## Overview
This project implements Dijkstra’s shortest path algorithm in three versions:
- Sequential
- Parallel using OpenMP
- Parallel using MPI

## Features
- Compare performance across different parallelization strategies
- Benchmark results included (`results_openmp.csv`, `mpi_results.txt`)
- Visual outputs (`output1.png`, `output2.png`, `output3.png`)

## Installation
```bash
gcc -fopenmp openmp_dijkstra.c -o openmp_dijkstra
mpicc mpi_openmp_dijkstra.c -o mpi_dijkstra

