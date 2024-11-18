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

### Map

Please see the screenshot [here](./assets/jit-map.png)

```bash
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (163)
------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                       |
        out: Storage,                                                               |
        out_shape: Shape,                                                           |
        out_strides: Strides,                                                       |
        in_storage: Storage,                                                        |
        in_shape: Shape,                                                            |
        in_strides: Strides,                                                        |
    ) -> None:                                                                      |
        # TODO: Implement for Task 3.1.                                             |
        # check if out, in are stride-aligned                                       |
        if out_strides == in_strides:-----------------------------------------------| #0
            for i in prange(len(out)):----------------------------------------------| #3
                out[i] = fn(in_storage[i])                                          |
        else:                                                                       |
            out_index = np.zeros(len(out_shape))  # buffer--------------------------| #1
            in_index = np.zeros(len(in_shape))  # buffer----------------------------| #2
            for i in prange(len(out)):----------------------------------------------| #4
                to_index(i, out_shape, out_index)                                   |
                broadcast_index(out_index, out_shape, in_shape, in_index)           |
                out[i] = fn(in_storage[index_to_position(in_index, in_strides)])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #0, #3, #1, #2, #4).
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
```

### Zip

Please see the screenshot [here](./assets/jit-zip.png)

```bash
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (210)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (210)
------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                       |
        out: Storage,                                                               |
        out_shape: Shape,                                                           |
        out_strides: Strides,                                                       |
        a_storage: Storage,                                                         |
        a_shape: Shape,                                                             |
        a_strides: Strides,                                                         |
        b_storage: Storage,                                                         |
        b_shape: Shape,                                                             |
        b_strides: Strides,                                                         |
    ) -> None:                                                                      |
        # TODO: Implement for Task 3.1.                                             |
        # check if out, a, b are stride-aligned                                     |
        if out_strides == a_strides and out_strides == b_strides:-------------------| #5, 6
            for i in prange(len(out)):----------------------------------------------| #10
                out[i] = fn(a_storage[i], b_storage[i])                             |
        else:                                                                       |
            out_index = np.zeros(len(out_shape))  # buffer--------------------------| #7
            a_index = np.zeros(len(a_shape))  # buffer------------------------------| #8
            b_index = np.zeros(len(b_shape))  # buffer------------------------------| #9
            for i in prange(len(out)):----------------------------------------------| #11
                to_index(i, out_shape, out_index)                                   |
                broadcast_index(out_index, out_shape, a_shape, a_index)             |
                broadcast_index(out_index, out_shape, b_shape, b_index)             |
                                                                                    |
                out[i] = fn(a_storage[index_to_position(a_index, a_strides)],       |
                            b_storage[index_to_position(b_index, b_strides)])       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 7 parallel for-
loop(s) (originating from loops labelled: #5, #6, #10, #7, #8, #9, #11).
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
```

### Reduce

Please see the screenshot [here](./assets/jit-reduce.png)

```bash
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (262)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (262)
---------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                 |
        out: Storage,                                                            |
        out_shape: Shape,                                                        |
        out_strides: Strides,                                                    |
        a_storage: Storage,                                                      |
        a_shape: Shape,                                                          |
        a_strides: Strides,                                                      |
        reduce_dim: int,                                                         |
    ) -> None:                                                                   |
        # TODO: Implement for Task 3.1.                                          |
        out_index = np.zeros(len(out_shape))  # buffer---------------------------| #12
        a_index = np.zeros(len(a_shape))  # buffer-------------------------------| #13
        for i in prange(len(out)):-----------------------------------------------| #14
            to_index(i, out_shape, out_index)                                    |
            # copy the out_index to the a_index (except for the reduce dim)      |
            for j in range(len(a_shape)):                                        |
                a_index[j if j < reduce_dim else j + 1] = out_index[j]           |
                                                                                 |
            a_index[reduce_dim] = 0                                              |
            a_pos = index_to_position(a_index, a_strides)                        |
            temp = a_storage[a_pos]  # avoid inner access to global variable     |
            for j in range(1, a_shape[reduce_dim]):                              |
                temp = fn(temp, a_storage[a_pos + j * a_strides[reduce_dim]])    |
            out[i] = temp                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #12, #13, #14).
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
```

### Matrix Multiply

Please see the screenshot [here](./assets/jit-matmul.png)

````bash
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (290)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /home/ubuntu/proj/mod3-Navxihziq/minitorch/fast_ops.py (290)
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
    for i in prange(len(out)):--------------------------------------------| #15
        # disassemble the index                                           |
        out_batch = i // (out_shape[-2] * out_shape[-1])                  |
        out_i = (i // out_shape[-1]) % out_shape[-2]                      |
        out_j = i % out_shape[-1]                                         |
                                                                          |
        a_pos = out_batch * a_batch_stride + out_i * a_strides[-2] + 0    |
        b_pos = out_batch * b_batch_stride + 0 + out_j * b_strides[-1]    |
                                                                          |
        acc = 0.0                                                         |
        for j in range(a_shape[-1]):    # iterate along the shared dim    |
            acc += (a_storage[a_pos + j * a_strides[-1]] *                |
                    b_storage[b_pos + j * b_strides[-2]])                 |
        out[i] = acc                                                      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #15).
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
