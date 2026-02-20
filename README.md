# mpiAllReduceCompressed

## Overview

This project implements and benchmarks the paper **"Accelerating MPI AllReduce Communication with Efficient GPU-Based Compression Schemes on Modern GPU Clusters"**.

## Implementation Scope

The project includes the following implementations and comparisons:

1. **Baseline Implementation**: Standard MPI AllReduce calls using standalone vendor implementations (native MPI)

2. **Routing Algorithm Implementations** (from scratch):
   - Ring topology algorithm
   - Recursive doubling algorithm

3. **Point-to-Point Compression**: Implementation of compression schemes for direct point-to-point communication

4. **Custom Implementation**: The paper's own optimized implementation incorporating GPU-based compression techniques

## Benchmarking

All implementations will be systematically benchmarked to compare:
- Communication latency
- Throughput
- Compression efficiency
- GPU utilization
- Overall performance gains compared to baseline
