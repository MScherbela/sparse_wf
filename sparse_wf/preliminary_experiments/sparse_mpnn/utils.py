import jax
import functools
import folx


def multi_vmap(f, n, in_axes=0):
    """Iteratively apply vmap n times."""
    for _ in range(n):
        f = jax.vmap(f, in_axes=in_axes)
    return f

def vmap_batch_dims(func, nr_non_batch_dims, in_axes=0):
    """Iteratively apply vmap over all batch dimensions.

    The number of batch-dims is inferred by specfiying the number of non-batch dims for each argument.
    At compile time, this function determines the nr of batch dims, checks consistency across arguments and applies the vmap
    
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # Get number of batch dimensions and check that it is consistent across arguments
        batch_dims = []
        for arg, non_batch in zip(args, nr_non_batch_dims):
            if non_batch is None:
                continue
            batch_dims.append(arg.ndim - non_batch)
        n_batch_dims = batch_dims[0]
        assert all(
            [bd == n_batch_dims for bd in batch_dims]
        ), "All arguments must have the same number of batch dimensions"

        # Vmap over all batch_dims
        vmapped_func = func
        for _ in range(n_batch_dims):
            vmapped_func = jax.vmap(vmapped_func, in_axes=in_axes)
        return vmapped_func(*args, **kwargs)

    return wrapped

def fwd_lap(f, argnums=None, sparsity_threshold=0.6):
    """Applies forward laplacian transform using the folx package, but addes the option to specifiy which args are being differentiated."""

    if argnums is None:
        # Take laplacian wrt to all arguments. This is the default of folx anyway so lets just use that
        return folx.forward_laplacian(f, sparsity_threshold=sparsity_threshold)

    if isinstance(argnums, int):
        argnums = (argnums,)
    argnums = sorted(argnums)
    assert len(set(argnums)) == len(argnums), "argnums must be unique"

    @functools.wraps(f)
    def transformed(*args):
        should_take_lap = [i in argnums for i in range(len(args))]

        # Create a new function that only depends on the argments that should be differentiated (specified by argnums)
        # and apply the forward laplacian transform to this function
        @functools.partial(folx.forward_laplacian, sparsity_threshold=sparsity_threshold)
        def func_with_only_args_to_diff(*args_to_diff_):
            # Combine the differentiable and non-differentiable arguments in their original order and pass them to the original function
            idx_arg_diff = 0
            combined_args = []
            for i, do_lap in enumerate(should_take_lap):
                if do_lap:
                    combined_args.append(args_to_diff_[idx_arg_diff])
                    idx_arg_diff += 1
                else:
                    combined_args.append(args[i])
            return f(*combined_args)

        args_to_diff = [arg for arg, do_lap in zip(args, should_take_lap) if do_lap]
        lap_array = func_with_only_args_to_diff(*args_to_diff)
        return lap_array

    return transformed

