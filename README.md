# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1

Please see the entire terminal buffer [here](./assets/parallel-check-buff.pdf)

### Map

```bash
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024 Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (163)
--------------------------------------------------------------------------------|loop #ID
    def _map(                                                                   |
        out: Storage,                                                           |
        out_shape: Shape,                                                       |
        out_strides: Strides,                                                   |
        in_storage: Storage,                                                    |
        in_shape: Shape,                                                        |
        in_strides: Strides,                                                    |
    ) -> None:                                                                  |
        # TODO: Implement for Task 3.1.                                         |
        # check if out, in are stride-aligned                                   |
        # if out_strides == in_strides:                                         |
        #     for i in prange(len(out)):                                        |
        #         out[i] = fn(in_storage[i])                                    |
        # else:                                                                 |
        # TODO: check if out, in are stride-aligned                             |
        # coerce the shape to int32                                             |
        out_shape = out_shape.astype(np.int32)                                  |
        in_shape = in_shape.astype(np.int32)                                    |
        for i in prange(len(out)):----------------------------------------------| #2
            out_index = np.zeros(len(out_shape), dtype=np.int32)  # buffer------| #0
            in_index = np.zeros(len(in_shape), dtype=np.int32)  # buffer--------| #1
            to_index(i, out_shape, out_index)                                   |
            broadcast_index(out_index, out_shape, in_shape, in_index)           |
            out[i] = fn(in_storage[index_to_position(in_index, in_strides)])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (182) is hoisted out of
the parallel loop labelled #2 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (183) is hoisted out of
the parallel loop labelled #2 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
None
```

### Zip

```bash
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024 Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (214)
------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                 |
        out: Storage,                                                         |
        out_shape: Shape,                                                     |
        out_strides: Strides,                                                 |
        a_storage: Storage,                                                   |
        a_shape: Shape,                                                       |
        a_strides: Strides,                                                   |
        b_storage: Storage,                                                   |
        b_shape: Shape,                                                       |
        b_strides: Strides,                                                   |
    ) -> None:                                                                |
        # TODO: Implement for Task 3.1.                                       |
        # coerce the shape to int32                                           |
        out_shape = out_shape.astype(np.int32)                                |
        a_shape = a_shape.astype(np.int32)                                    |
        b_shape = b_shape.astype(np.int32)                                    |
        # TODO: check if out, a, b are stride-aligned                         |
        # if (                                                                |
        #     len(out_shape) == len(a_shape) == len(b_shape)                  |
        #     and np.array_equal(out_shape, a_shape)                          |
        #     and np.array_equal(out_shape, b_shape)                          |
        #     and np.array_equal(out_strides, a_strides)                      |
        #     and np.array_equal(out_strides, b_strides)                      |
        # ):                                                                  |
        #     for i in prange(len(out)):                                      |
        #         out[i] = fn(a_storage[i], b_storage[i])                     |
        # else:                                                               |
        for i in prange(len(out)):--------------------------------------------| #6
            out_index = np.zeros(len(out_shape), dtype=np.int32)  # buffer----| #3
            a_index = np.zeros(len(a_shape), dtype=np.int32)  # buffer--------| #4
            b_index = np.zeros(len(b_shape), dtype=np.int32)  # buffer--------| #5
            to_index(i, out_shape, out_index)                                 |
            broadcast_index(out_index, out_shape, a_shape, a_index)           |
            broadcast_index(out_index, out_shape, b_shape, b_index)           |
                                                                              |
            out[i] = fn(                                                      |
                a_storage[index_to_position(a_index, a_strides)],             |
                b_storage[index_to_position(b_index, b_strides)],             |
            )                                                                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #6, #3, #4, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)



Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (242) is hoisted out of
the parallel loop labelled #6 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (243) is hoisted out of
the parallel loop labelled #6 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (244) is hoisted out of
the parallel loop labelled #6 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
None
```

### Reduce

```bash
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (279)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024 Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (279)
----------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                        |
        out: Storage,                                                                   |
        out_shape: Shape,                                                               |
        out_strides: Strides,                                                           |
        a_storage: Storage,                                                             |
        a_shape: Shape,                                                                 |
        a_strides: Strides,                                                             |
        reduce_dim: int,                                                                |
    ) -> None:                                                                          |
        # TODO: Implement for Task 3.1.                                                 |
        # coerce the shape to int32                                                     |
        out_shape = out_shape.astype(np.int32)                                          |
        a_shape = a_shape.astype(np.int32)                                              |
        for i in prange(len(out)):------------------------------------------------------| #9
            out_index = np.zeros(len(out_shape), dtype=np.int32)  # buffer--------------| #7
            a_index = np.zeros(len(a_shape), dtype=np.int32)  # buffer------------------| #8
            to_index(i, out_shape, out_index)                                           |
            # copy the out_index to the a_index (except for the reduce dim)             |
            for j in range(len(a_shape) - 1):                                           |
                j = j if j < reduce_dim else j + 1                                      |
                a_index[j] = out_index[j]                                               |
                                                                                        |
            a_index[reduce_dim] = 0                                                     |
            a_pos = index_to_position(a_index, a_strides)                               |
            temp: float = a_storage[a_pos]  # avoid inner access to global variable     |
            for j in range(1, a_shape[reduce_dim]):                                     |
                temp = fn(temp, float(a_storage[a_pos + j * a_strides[reduce_dim]]))    |
            out[i] = temp                                                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #9, #7, #8).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--9 is a parallel loop
   +--8 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--8 (serial)
   +--7 (serial)



Parallel region 0 (loop #9) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (293) is hoisted out of
the parallel loop labelled #9 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (294) is hoisted out of
the parallel loop labelled #9 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)  # buffer
    - numpy.empty() is used for the allocation.
None
```

### Matrix Multiply

````bash
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024
Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (311)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/qizhixuan/Library/CloudStorage/OneDrive-Personal/2024 Fall/MLE/workspace/mod3-Navxihziq/minitorch/fast_ops.py (311)
--------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                              |
    out: Storage,                                                         |
    out_shape: Shape,                                                     |
    out_strides: Strides,                                                 |
    a_storage: Storage,                                                   |
    a_shape: Shape,                                                       |
    a_strides: Strides,                                                   |
    b_storage: Storage,                                                   |
    b_shape: Shape,                                                       |
    b_strides: Strides,                                                   |
) -> None:                                                                |
    """NUMBA tensor matrix multiply function.                             |
                                                                          |
    Should work for any tensor shapes that broadcast as long as           |
                                                                          |
    ```                                                                   |
    assert a_shape[-1] == b_shape[-2]                                     |
    ```                                                                   |
                                                                          |
    Optimizations:                                                        |
                                                                          |
    * Outer loop in parallel                                              |
    * No index buffers or function calls                                  |
    * Inner loop should have no global writes, 1 multiply.                |
                                                                          |
                                                                          |
    Args:                                                                 |
    ----                                                                  |
        out (Storage): storage for `out` tensor                           |
        out_shape (Shape): shape for `out` tensor                         |
        out_strides (Strides): strides for `out` tensor                   |
        a_storage (Storage): storage for `a` tensor                       |
        a_shape (Shape): shape for `a` tensor                             |
        a_strides (Strides): strides for `a` tensor                       |
        b_storage (Storage): storage for `b` tensor                       |
        b_shape (Shape): shape for `b` tensor                             |
        b_strides (Strides): strides for `b` tensor                       |
                                                                          |
    Returns:                                                              |
    -------                                                               |
        None : Fills in `out`                                             |
                                                                          |
    """                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                |
                                                                          |
    # TODO: Implement for Task 3.2.                                       |
    for i in prange(len(out)):--------------------------------------------| #10
        # disassemble the index                                           |
        out_batch = i // (out_shape[-2] * out_shape[-1])                  |
        out_j = (i % out_strides[0]) % out_shape[-1]                      |
        out_i = (i % out_strides[0]) // out_shape[-1]                     |
                                                                          |
        a_pos = out_batch * a_batch_stride + out_i * a_strides[-2] + 0    |
        b_pos = out_batch * b_batch_stride + 0 + out_j * b_strides[-1]    |
                                                                          |
        acc = 0.0                                                         |
        for j in range(a_shape[-1]):  # iterate along the shared dim      |
            a_location = a_pos + j * a_strides[-1]                        |
            b_location = b_pos + j * b_strides[-2]                        |
            acc += a_storage[a_location] * b_storage[b_location]          |
        out[i] = acc                                                      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
````

## Task 3.5

### GPU

#### Simple Dataset

Please see [file](./assets/simple-gpu.pdf) for entire log.

**Average Time per Epoch**: 1.91s

```bash
Epoch 0 loss 6.704483083841267 correct 31 time 3.92s time per epoch 3.92s
Epoch 10 loss 4.216760860538249 correct 45 time 19.04s time per epoch 1.73s
Epoch 20 loss 2.039506239455864 correct 50 time 34.12s time per epoch 1.62s
Epoch 30 loss 2.6713099276871577 correct 47 time 49.13s time per epoch 1.58s
Epoch 40 loss 0.5812707290991249 correct 49 time 64.17s time per epoch 1.57s
Epoch 50 loss 1.0392651753198405 correct 50 time 79.20s time per epoch 1.55s
Epoch 60 loss 1.1209911862628306 correct 49 time 94.28s time per epoch 1.55s
Epoch 70 loss 1.4330156153269031 correct 48 time 109.39s time per epoch 1.54s
Epoch 80 loss 0.6971472984702927 correct 50 time 124.45s time per epoch 1.54s
Epoch 90 loss 0.06328335212521252 correct 48 time 139.66s time per epoch 1.53s
Epoch 100 loss 1.7030878306924464 correct 48 time 154.77s time per epoch 1.53s
Epoch 110 loss 0.7518350296464371 correct 50 time 169.90s time per epoch 1.53s
Epoch 120 loss 0.7894673939097282 correct 50 time 184.97s time per epoch 1.53s
Epoch 130 loss 0.4187893076324422 correct 50 time 200.24s time per epoch 1.53s
Epoch 140 loss 0.9183523345697427 correct 50 time 215.51s time per epoch 1.53s
Epoch 150 loss 0.4058593412944356 correct 49 time 230.97s time per epoch 1.53s
Epoch 160 loss 0.35085626577418594 correct 50 time 246.40s time per epoch 1.53s
Epoch 170 loss 0.20101432823167287 correct 50 time 261.89s time per epoch 1.53s
Epoch 180 loss 1.072228514334062 correct 50 time 277.31s time per epoch 1.53s
Epoch 190 loss 0.4388051777287315 correct 50 time 292.34s time per epoch 1.53s
Epoch 200 loss 0.7136973971679863 correct 50 time 307.34s time per epoch 1.53s
Epoch 210 loss 0.025048719554116642 correct 50 time 322.33s time per epoch 1.53s
Epoch 220 loss 0.0611063990383387 correct 50 time 337.46s time per epoch 1.53s
Epoch 230 loss 0.0977671856627235 correct 50 time 352.54s time per epoch 1.53s
Epoch 240 loss 0.26556086681306945 correct 50 time 368.65s time per epoch 1.53s
Epoch 250 loss 0.5674033877365865 correct 50 time 383.93s time per epoch 1.53s
Epoch 260 loss 1.0000835721920882 correct 50 time 398.63s time per epoch 1.53s
Epoch 270 loss 1.2565762487285232 correct 50 time 412.67s time per epoch 1.52s
Epoch 280 loss 1.3230792099051627 correct 50 time 426.86s time per epoch 1.52s
Epoch 290 loss 0.14563086657169969 correct 50 time 440.96s time per epoch 1.52s
Epoch 300 loss 0.14459728006448896 correct 50 time 454.75s time per epoch 1.51s
Epoch 310 loss 0.8089563099364643 correct 50 time 468.54s time per epoch 1.51s
Epoch 320 loss 0.18935787933598136 correct 50 time 482.53s time per epoch 1.50s
Epoch 330 loss 1.1890661830555347 correct 49 time 500.71s time per epoch 1.51s
Epoch 340 loss 0.49201291101053174 correct 50 time 528.45s time per epoch 1.55s
Epoch 350 loss 0.007754055644567672 correct 50 time 556.52s time per epoch 1.59s
Epoch 360 loss 0.0050600031739717844 correct 50 time 595.16s time per epoch 1.65s
Epoch 370 loss 0.2194503771370788 correct 50 time 637.53s time per epoch 1.72s
Epoch 380 loss 0.06837081958231209 correct 50 time 680.27s time per epoch 1.79s
Epoch 390 loss 0.12990114798816485 correct 50 time 723.25s time per epoch 1.85s
Epoch 400 loss 0.26557950979150396 correct 50 time 764.03s time per epoch 1.91s
```

#### Split Dataset

Please see [file](./assets/split-gpu.pdf) for entire log.

**Average Time per Epoch**: 1.46s

```bash
Epoch 0 loss 7.380800933932704 correct 38 time 3.31s time per epoch 3.31s
Epoch 10 loss 5.085222153062372 correct 39 time 16.88s time per epoch 1.53s
Epoch 20 loss 3.835299699035385 correct 40 time 30.46s time per epoch 1.45s
Epoch 30 loss 3.190248188743328 correct 44 time 43.98s time per epoch 1.42s
Epoch 40 loss 3.8206068822429513 correct 47 time 57.60s time per epoch 1.40s
Epoch 50 loss 3.374021549755926 correct 48 time 71.43s time per epoch 1.40s
Epoch 60 loss 2.3818731339270083 correct 49 time 85.20s time per epoch 1.40s
Epoch 70 loss 1.6159477534116222 correct 46 time 98.98s time per epoch 1.39s
Epoch 80 loss 2.62678262333655 correct 49 time 112.70s time per epoch 1.39s
Epoch 90 loss 2.3393019368962915 correct 49 time 126.44s time per epoch 1.39s
Epoch 100 loss 1.7873210729655682 correct 50 time 140.18s time per epoch 1.39s
Epoch 110 loss 1.074079664942238 correct 50 time 153.97s time per epoch 1.39s
Epoch 120 loss 1.759821848904663 correct 50 time 167.73s time per epoch 1.39s
Epoch 130 loss 0.6410341844847615 correct 50 time 181.50s time per epoch 1.39s
Epoch 140 loss 0.42725729411551483 correct 50 time 195.25s time per epoch 1.38s
Epoch 150 loss 0.3883554718120374 correct 50 time 209.04s time per epoch 1.38s
Epoch 160 loss 0.6721461643509723 correct 50 time 223.20s time per epoch 1.39s
Epoch 170 loss 0.5742892264020044 correct 50 time 238.27s time per epoch 1.39s
Epoch 180 loss 0.8319282409943765 correct 50 time 253.28s time per epoch 1.40s
Epoch 190 loss 0.49926423953554727 correct 50 time 268.29s time per epoch 1.40s
Epoch 200 loss 0.6805619185615739 correct 50 time 283.77s time per epoch 1.41s
Epoch 210 loss 1.0074121208991353 correct 50 time 299.20s time per epoch 1.42s
Epoch 220 loss 0.3680246122168671 correct 50 time 314.54s time per epoch 1.42s
Epoch 230 loss 0.47463329277364064 correct 50 time 329.56s time per epoch 1.43s
Epoch 240 loss 0.6123216199014468 correct 50 time 344.68s time per epoch 1.43s
Epoch 250 loss 0.5780413248770209 correct 50 time 359.92s time per epoch 1.43s
Epoch 260 loss 0.8160247193127806 correct 50 time 374.99s time per epoch 1.44s
Epoch 270 loss 0.754561749733112 correct 50 time 390.08s time per epoch 1.44s
Epoch 280 loss 0.15056632163918357 correct 50 time 405.18s time per epoch 1.44s
Epoch 290 loss 0.21470546061614496 correct 50 time 420.45s time per epoch 1.44s
Epoch 300 loss 0.4210963156368792 correct 50 time 435.58s time per epoch 1.45s
Epoch 310 loss 0.19411553162400305 correct 50 time 450.52s time per epoch 1.45s
Epoch 320 loss 0.4964319201860512 correct 50 time 465.57s time per epoch 1.45s
Epoch 330 loss 0.4999424263761936 correct 50 time 480.61s time per epoch 1.45s
Epoch 340 loss 0.3308044029318727 correct 50 time 495.79s time per epoch 1.45s
Epoch 350 loss 0.27953803901065816 correct 50 time 511.03s time per epoch 1.46s
Epoch 360 loss 0.6042746298198678 correct 50 time 526.04s time per epoch 1.46s
Epoch 370 loss 0.5785738079168321 correct 50 time 541.20s time per epoch 1.46s
Epoch 380 loss 0.383365801580663 correct 50 time 556.49s time per epoch 1.46s
Epoch 390 loss 0.3241101114025531 correct 50 time 571.58s time per epoch 1.46s
Epoch 400 loss 0.49098038753028084 correct 50 time 586.69s time per epoch 1.46s
```

#### XOR Dataset

Please see [file](./assets/xor-gpu.pdf) for the entire log.

**Average Time per Epoch**: 1.49s

```bash
Epoch 0 loss 7.929135135737758 correct 29 time 3.79s time per epoch 3.79s
Epoch 10 loss 4.552853840425255 correct 39 time 18.11s time per epoch 1.65s
Epoch 20 loss 4.036263637462243 correct 46 time 32.44s time per epoch 1.54s
Epoch 30 loss 5.893117552297534 correct 42 time 46.35s time per epoch 1.50s
Epoch 40 loss 2.4576929605961153 correct 46 time 60.18s time per epoch 1.47s
Epoch 50 loss 1.535034618598493 correct 48 time 74.00s time per epoch 1.45s
Epoch 60 loss 3.2892836280156157 correct 48 time 88.12s time per epoch 1.44s
Epoch 70 loss 3.251929496623473 correct 46 time 101.94s time per epoch 1.44s
Epoch 80 loss 1.9497109010185258 correct 48 time 115.79s time per epoch 1.43s
Epoch 90 loss 1.5444758783124537 correct 48 time 129.63s time per epoch 1.42s
Epoch 100 loss 1.1125971672677224 correct 49 time 143.46s time per epoch 1.42s
Epoch 110 loss 3.682435553173611 correct 46 time 157.32s time per epoch 1.42s
Epoch 120 loss 1.7617037055094134 correct 49 time 171.43s time per epoch 1.42s
Epoch 130 loss 1.973978643164274 correct 48 time 186.53s time per epoch 1.42s
Epoch 140 loss 0.6202532269373142 correct 49 time 201.80s time per epoch 1.43s
Epoch 150 loss 1.1928340105537187 correct 49 time 217.35s time per epoch 1.44s
Epoch 160 loss 0.4595514254053154 correct 49 time 232.68s time per epoch 1.45s
Epoch 170 loss 0.6903858274016108 correct 49 time 247.76s time per epoch 1.45s
Epoch 180 loss 0.4490669854618037 correct 49 time 262.98s time per epoch 1.45s
Epoch 190 loss 1.2440319869789267 correct 50 time 278.50s time per epoch 1.46s
Epoch 200 loss 0.3069011765969647 correct 49 time 293.75s time per epoch 1.46s
Epoch 210 loss 0.8285602489490569 correct 50 time 309.02s time per epoch 1.46s
Epoch 220 loss 0.7603352541054381 correct 50 time 324.18s time per epoch 1.47s
Epoch 230 loss 0.2026691534503689 correct 50 time 339.34s time per epoch 1.47s
Epoch 240 loss 0.8837380207213554 correct 50 time 354.51s time per epoch 1.47s
Epoch 250 loss 0.6624675861406382 correct 50 time 369.80s time per epoch 1.47s
Epoch 260 loss 0.27289019807549864 correct 50 time 385.29s time per epoch 1.48s
Epoch 270 loss 0.4461875158112574 correct 50 time 400.77s time per epoch 1.48s
Epoch 280 loss 0.09489040850543254 correct 50 time 416.33s time per epoch 1.48s
Epoch 290 loss 0.4086393045778214 correct 50 time 431.48s time per epoch 1.48s
Epoch 300 loss 0.18219400200469532 correct 50 time 446.75s time per epoch 1.48s
Epoch 310 loss 0.5106873530330133 correct 50 time 462.13s time per epoch 1.49s
Epoch 320 loss 0.45995888113596783 correct 50 time 477.67s time per epoch 1.49s
Epoch 330 loss 0.1200706412667914 correct 50 time 493.10s time per epoch 1.49s
Epoch 340 loss 0.23290781846236983 correct 50 time 508.35s time per epoch 1.49s
Epoch 350 loss 0.20696767158308954 correct 50 time 523.53s time per epoch 1.49s
Epoch 360 loss 0.24870764326832792 correct 50 time 538.91s time per epoch 1.49s
Epoch 370 loss 0.27658419721677224 correct 50 time 554.12s time per epoch 1.49s
Epoch 380 loss 0.3101088048969413 correct 50 time 568.82s time per epoch 1.49s
Epoch 390 loss 0.17399900305812901 correct 50 time 582.94s time per epoch 1.49s
Epoch 400 loss 0.25045310158620887 correct 50 time 596.92s time per epoch 1.49s
```

### CPU

#### Simple Dataset

Please see [file](./assets/simple-cpu.pdf) for the entire log.

**Average Time per Epoch**: 0.08s

```bash
Epoch  0  loss  5.212761571257698 correct 44 time 11.24s time per epoch 11.24s
Epoch  10  loss  2.772970757638876 correct 49 time 11.83s time per epoch 1.08s
Epoch  20  loss  0.6360609350496206 correct 49 time 12.41s time per epoch 0.59s
Epoch  30  loss  0.8474514549184928 correct 50 time 12.98s time per epoch 0.42s
Epoch  40  loss  0.08597348255800996 correct 50 time 13.56s time per epoch 0.33s
Epoch  50  loss  0.14994450363395617 correct 50 time 14.13s time per epoch 0.28s
Epoch  60  loss  0.628109536977768 correct 50 time 14.72s time per epoch 0.24s
Epoch  70  loss  0.6136681035790843 correct 50 time 15.31s time per epoch 0.22s
Epoch  80  loss  0.4646874546380364 correct 50 time 15.88s time per epoch 0.20s
Epoch  90  loss  0.39231921731916564 correct 49 time 16.46s time per epoch 0.18s
Epoch  100  loss  0.5182912253522517 correct 50 time 17.02s time per epoch 0.17s
Epoch  110  loss  0.5721051549644813 correct 50 time 17.59s time per epoch 0.16s
Epoch  120  loss  0.8436973421915123 correct 49 time 18.17s time per epoch 0.15s
Epoch  130  loss  0.5074232538068905 correct 50 time 18.74s time per epoch 0.14s
Epoch  140  loss  0.012085385266606427 correct 50 time 19.30s time per epoch 0.14s
Epoch  150  loss  0.011297120566273227 correct 50 time 19.88s time per epoch 0.13s
Epoch  160  loss  0.6591053493692443 correct 50 time 20.47s time per epoch 0.13s
Epoch  170  loss  0.23920266566688947 correct 50 time 21.05s time per epoch 0.12s
Epoch  180  loss  0.5250741233513093 correct 50 time 21.62s time per epoch 0.12s
Epoch  190  loss  0.21752122652981606 correct 50 time 22.20s time per epoch 0.12s
Epoch  200  loss  0.3328125312395841 correct 50 time 22.77s time per epoch 0.11s
Epoch  210  loss  0.008738172855092546 correct 50 time 23.34s time per epoch 0.11s
Epoch  220  loss  0.09479674245558851 correct 50 time 23.92s time per epoch 0.11s
Epoch  230  loss  0.5081987298741725 correct 50 time 24.51s time per epoch 0.11s
Epoch  240  loss  0.13894768511135583 correct 50 time 25.08s time per epoch 0.10s
Epoch  250  loss  0.28314040604243035 correct 50 time 25.66s time per epoch 0.10s
Epoch  260  loss  0.43959681427535874 correct 50 time 26.23s time per epoch 0.10s
Epoch  270  loss  0.16878941069768733 correct 50 time 26.82s time per epoch 0.10s
Epoch  280  loss  0.5037562006605987 correct 50 time 27.39s time per epoch 0.10s
Epoch  290  loss  0.011219989893343038 correct 50 time 27.97s time per epoch 0.10s
Epoch  300  loss  0.23696520805854893 correct 50 time 28.59s time per epoch 0.09s
Epoch  310  loss  0.1593568470069938 correct 50 time 29.17s time per epoch 0.09s
Epoch  320  loss  0.45481373047405044 correct 50 time 29.74s time per epoch 0.09s
Epoch  330  loss  0.4306181481709963 correct 50 time 30.32s time per epoch 0.09s
Epoch  340  loss  0.04346007028388669 correct 50 time 30.89s time per epoch 0.09s
Epoch  350  loss  0.07366034645170304 correct 50 time 31.47s time per epoch 0.09s
Epoch  360  loss  0.3174855915995364 correct 50 time 32.04s time per epoch 0.09s
Epoch  370  loss  0.05087934011473569 correct 50 time 32.63s time per epoch 0.09s
Epoch  380  loss  0.3039275843356951 correct 50 time 33.20s time per epoch 0.09s
Epoch  390  loss  0.35956697626862305 correct 50 time 33.77s time per epoch 0.09s
Epoch  400  loss  0.3019835045930892 correct 50 time 34.34s time per epoch 0.09s
Epoch  410  loss  0.051246287377090124 correct 50 time 34.91s time per epoch 0.08s
Epoch  420  loss  0.24582857910359326 correct 50 time 35.48s time per epoch 0.08s
Epoch  430  loss  0.3137192977271656 correct 50 time 36.05s time per epoch 0.08s
Epoch  440  loss  0.11873330220584688 correct 50 time 36.62s time per epoch 0.08s
Epoch  450  loss  0.0056283467905821685 correct 50 time 37.18s time per epoch 0.08s
Epoch  460  loss  0.01396355651906908 correct 50 time 37.75s time per epoch 0.08s
Epoch  470  loss  0.02939271021725448 correct 50 time 38.32s time per epoch 0.08s
Epoch  480  loss  0.1605528973096863 correct 50 time 38.89s time per epoch 0.08s
Epoch  490  loss  0.026377585818314595 correct 50 time 39.47s time per epoch 0.08s
```

#### Split Dataset

Please see [file](./assets/split-cpu.pdf) for the entire log.

**Average Time per Epoch**: 1.46s

```bash
Epoch 0 loss 7.380800933932704 correct 38 time 3.31s time per epoch 3.31s
Epoch 10 loss 5.085222153062372 correct 39 time 16.88s time per epoch 1.53s
Epoch 20 loss 3.835299699035385 correct 40 time 30.46s time per epoch 1.45s
Epoch 30 loss 3.190248188743328 correct 44 time 43.98s time per epoch 1.42s
Epoch 40 loss 3.8206068822429513 correct 47 time 57.60s time per epoch 1.40s
Epoch 50 loss 3.374021549755926 correct 48 time 71.43s time per epoch 1.40s
Epoch 60 loss 2.3818731339270083 correct 49 time 85.20s time per epoch 1.40s
Epoch 70 loss 1.6159477534116222 correct 46 time 98.98s time per epoch 1.39s
Epoch 80 loss 2.62678262333655 correct 49 time 112.70s time per epoch 1.39s
Epoch 90 loss 2.3393019368962915 correct 49 time 126.44s time per epoch 1.39s
Epoch 100 loss 1.7873210729655682 correct 50 time 140.18s time per epoch 1.39s
Epoch 110 loss 1.074079664942238 correct 50 time 153.97s time per epoch 1.39s
Epoch 120 loss 1.759821848904663 correct 50 time 167.73s time per epoch 1.39s
Epoch 130 loss 0.6410341844847615 correct 50 time 181.50s time per epoch 1.39s
Epoch 140 loss 0.42725729411551483 correct 50 time 195.25s time per epoch 1.38s
Epoch 150 loss 0.3883554718120374 correct 50 time 209.04s time per epoch 1.38s
Epoch 160 loss 0.6721461643509723 correct 50 time 223.20s time per epoch 1.39s
Epoch 170 loss 0.5742892264020044 correct 50 time 238.27s time per epoch 1.39s
Epoch 180 loss 0.8319282409943765 correct 50 time 253.28s time per epoch 1.40s
Epoch 190 loss 0.49926423953554727 correct 50 time 268.29s time per epoch 1.40s
Epoch 200 loss 0.6805619185615739 correct 50 time 283.77s time per epoch 1.41s
Epoch 210 loss 1.0074121208991353 correct 50 time 299.20s time per epoch 1.42s
Epoch 220 loss 0.3680246122168671 correct 50 time 314.54s time per epoch 1.42s
Epoch 230 loss 0.47463329277364064 correct 50 time 329.56s time per epoch 1.43s
Epoch 240 loss 0.6123216199014468 correct 50 time 344.68s time per epoch 1.43s
Epoch 250 loss 0.5780413248770209 correct 50 time 359.92s time per epoch 1.43s
Epoch 260 loss 0.8160247193127806 correct 50 time 374.99s time per epoch 1.44s
Epoch 270 loss 0.754561749733112 correct 50 time 390.08s time per epoch 1.44s
Epoch 280 loss 0.15056632163918357 correct 50 time 405.18s time per epoch 1.44s
Epoch 290 loss 0.21470546061614496 correct 50 time 420.45s time per epoch 1.44s
Epoch 300 loss 0.4210963156368792 correct 50 time 435.58s time per epoch 1.45s
Epoch 310 loss 0.19411553162400305 correct 50 time 450.52s time per epoch 1.45s
Epoch 320 loss 0.4964319201860512 correct 50 time 465.57s time per epoch 1.45s
Epoch 330 loss 0.4999424263761936 correct 50 time 480.61s time per epoch 1.45s
Epoch 340 loss 0.3308044029318727 correct 50 time 495.79s time per epoch 1.45s
Epoch 350 loss 0.27953803901065816 correct 50 time 511.03s time per epoch 1.46s
Epoch 360 loss 0.6042746298198678 correct 50 time 526.04s time per epoch 1.46s
Epoch 370 loss 0.5785738079168321 correct 50 time 541.20s time per epoch 1.46s
Epoch 380 loss 0.383365801580663 correct 50 time 556.49s time per epoch 1.46s
Epoch 390 loss 0.3241101114025531 correct 50 time 571.58s time per epoch 1.46s
Epoch 400 loss 0.49098038753028084 correct 50 time 586.69s time per epoch 1.46s
```

#### XOR Dataset

Please see [file](./assets/xor-cpu.pdf) for the entire log.

**Average Time per Epoch**: 3.84s

```bash
Epoch  0  loss  5.754644474358288 correct 38 time 32.90s time per epoch 32.90s
Epoch  10  loss  4.668400473097102 correct 44 time 163.09s time per epoch 14.83s
Epoch  20  loss  4.186727796901085 correct 39 time 287.90s time per epoch 13.71s
Epoch  30  loss  2.368754801666736 correct 44 time 459.23s time per epoch 14.81s
Epoch  40  loss  2.93651503127627 correct 43 time 637.10s time per epoch 15.54s
Epoch  50  loss  2.767497908824362 correct 46 time 800.40s time per epoch 15.69s
Epoch  60  loss  2.883728761749026 correct 47 time 937.66s time per epoch 15.37s
Epoch  70  loss  3.599298356614441 correct 45 time 1082.65s time per epoch 15.25s
Epoch  80  loss  4.630686163809956 correct 44 time 1265.51s time per epoch 15.62s
Epoch  90  loss  1.5896193515560675 correct 47 time 1431.61s time per epoch 15.73s
Epoch  100  loss  1.0845923931480963 correct 45 time 1542.42s time per epoch 15.27s
Epoch  110  loss  2.0592190242254644 correct 46 time 1605.49s time per epoch 14.46s
Epoch  120  loss  4.080515545245338 correct 46 time 1673.11s time per epoch 13.83s
Epoch  130  loss  1.6710399269940734 correct 47 time 1739.91s time per epoch 13.28s
Epoch  140  loss  2.4290992359368433 correct 47 time 1802.28s time per epoch 12.78s
Epoch  150  loss  2.621350065889716 correct 48 time 1856.94s time per epoch 12.30s
Epoch  160  loss  1.6262897653763002 correct 48 time 1864.92s time per epoch 11.58s
Epoch  170  loss  1.7120860103780235 correct 47 time 1865.50s time per epoch 10.91s
Epoch  180  loss  2.8526594307144286 correct 47 time 1866.07s time per epoch 10.31s
Epoch  190  loss  1.2219833369771198 correct 47 time 1866.65s time per epoch 9.77s
Epoch  200  loss  0.8459799766361761 correct 48 time 1867.22s time per epoch 9.29s
Epoch  210  loss  1.5358092897054958 correct 47 time 1867.80s time per epoch 8.85s
Epoch  220  loss  2.394709260487286 correct 47 time 1868.37s time per epoch 8.45s
Epoch  230  loss  3.4884482310512106 correct 48 time 1868.95s time per epoch 8.09s
Epoch  240  loss  1.7850805400288865 correct 47 time 1869.52s time per epoch 7.76s
Epoch  250  loss  1.5664356352639803 correct 48 time 1870.10s time per epoch 7.45s
Epoch  260  loss  0.9277176434978458 correct 47 time 1870.67s time per epoch 7.17s
Epoch  270  loss  3.6335612788588625 correct 43 time 1871.25s time per epoch 6.90s
Epoch  280  loss  0.6740529377526386 correct 47 time 1871.82s time per epoch 6.66s
Epoch  290  loss  1.3462468279356623 correct 48 time 1872.40s time per epoch 6.43s
Epoch  300  loss  1.2819988161359424 correct 47 time 1873.01s time per epoch 6.22s
Epoch  310  loss  0.30934023679710415 correct 48 time 1873.58s time per epoch 6.02s
Epoch  320  loss  0.5730039262682568 correct 48 time 1874.16s time per epoch 5.84s
Epoch  330  loss  3.141696974442069 correct 48 time 1874.74s time per epoch 5.66s
Epoch  340  loss  1.6656871984143224 correct 45 time 1875.32s time per epoch 5.50s
Epoch  350  loss  0.9522340918295624 correct 48 time 1875.89s time per epoch 5.34s
Epoch  360  loss  1.3509611464609221 correct 49 time 1876.46s time per epoch 5.20s
Epoch  370  loss  1.6823132352355032 correct 48 time 1877.04s time per epoch 5.06s
Epoch  380  loss  0.28935126952880974 correct 48 time 1877.62s time per epoch 4.93s
Epoch  390  loss  2.7017300202556402 correct 49 time 1878.19s time per epoch 4.80s
Epoch  400  loss  1.4929511628561354 correct 49 time 1878.76s time per epoch 4.69s
Epoch  410  loss  1.0444300969019389 correct 48 time 1879.34s time per epoch 4.57s
Epoch  420  loss  2.4890737490472996 correct 49 time 1879.92s time per epoch 4.47s
Epoch  430  loss  2.2849609168615848 correct 49 time 1880.49s time per epoch 4.36s
Epoch  440  loss  0.9742376367495362 correct 49 time 1881.06s time per epoch 4.27s
Epoch  450  loss  0.061624338774118506 correct 49 time 1881.63s time per epoch 4.17s
Epoch  460  loss  0.11101031719580752 correct 49 time 1882.21s time per epoch 4.08s
Epoch  470  loss  0.3314695916593982 correct 49 time 1882.78s time per epoch 4.00s
Epoch  480  loss  1.0817226728171065 correct 49 time 1883.35s time per epoch 3.92s
Epoch  490  loss  2.3290008785945524 correct 49 time 1883.93s time per epoch 3.84s
```

### Larger Model

500 Layers on Simple Dataset

Please see [file](./assets/simple-500.pdf) for the entire log.

**Average Time per Epoch**: 16.88s

```bash
Epoch  0  loss  4.014456941608819 correct 35 time 51.21s time per epoch 51.21s
Epoch  10  loss  2.038388310489951 correct 49 time 229.52s time per epoch 20.87s
Epoch  20  loss  0.5117186043736055 correct 49 time 407.52s time per epoch 19.41s
Epoch  30  loss  0.9141888765023622 correct 50 time 559.71s time per epoch 18.06s
Epoch  40  loss  0.4470544204201721 correct 50 time 696.43s time per epoch 16.99s
Epoch  50  loss  0.24005576617214958 correct 50 time 856.94s time per epoch 16.80s
Epoch  60  loss  0.3725644143611874 correct 50 time 1039.47s time per epoch 17.04s
Epoch  70  loss  0.01381404037992649 correct 50 time 1198.31s time per epoch 16.88s
```
