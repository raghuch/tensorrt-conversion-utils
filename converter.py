# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa: E501


import argparse
from datetime import datetime
import sys

import tensorrt as trt

from polygraphy.logger import G_LOGGER as PG_LOGGER

# NOTE: please refer to each corresponding class
# to see actual definition of each function.
# For more details, please read a following guide.
# https://github.com/NVIDIA/TensorRT/blob/release/8.2/tools/Polygraphy/polygraphy/README.md#loaders
from polygraphy.backend.trt import (
    network_from_onnx_path,
    engine_from_network,
    save_engine,
)

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
)


def print_with_datetime(msg):
    print(f"{datetime.now()} {msg}")


def add_extra_fp32(network):
    """
    Force operations involved in layer norm to run in FP32 precision.

    This function is copied from an official example code below.
    https://github.com/NVIDIA/TensorRT/blob/73a7cccd6224035d9008cdde5bc35db77212c402/demo/HuggingFace/T5/export.py#L51
    """
    pow_ops = {}
    for layer_index, layer in enumerate(network):
        if layer.type == trt.LayerType.IDENTITY:
            all_fp32 = all(
                layer.output_type_is_set(o)
                and layer.get_output_type(o) == trt.float32
                for o in range(layer.num_outputs)
            )
            if all_fp32:
                if layer.get_input(0).dtype == trt.float32:
                    layer.precision = trt.float32

        if layer.type == trt.LayerType.ELEMENTWISE:
            layer.__class__ = getattr(trt, "IElementWiseLayer")
            if layer.op == trt.ElementWiseOperation.POW:
                pow_ops[layer] = layer_index
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

    for _, index in pow_ops.items():
        # Iterate from few layers before pow to include
        # residual add and cast op.
        # Iterate till 10 layers after pow op to include
        # all operations included in layer norm.
        START_OFFSET = 4
        END_OFFSET = 12
        for i in range(index - START_OFFSET, index + END_OFFSET):
            l = network.get_layer(i)  # noqa: E741
            if l.type == trt.LayerType.REDUCE:
                l.precision = trt.float32
                l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.ELEMENTWISE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.SUM:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.UNARY:
                l.__class__ = getattr(trt, "IUnaryLayer")
                if l.op == trt.UnaryOperation.SQRT:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.ELEMENTWISE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.DIV:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.ELEMENTWISE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.PROD:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

    return network


def fix_fp16_network(network_definition, logger):
    producers = {}
    processed = {}
    fixed_layers = 0
    fixed_outputs = 0

    def get_producers(network_definition):
        producers = {}
        for layer in network_definition:
            for outnum in range(layer.num_outputs):
                name = layer.get_output(outnum).name
                if name is not None:  # network input
                    if name in producers:
                        logger.warning(f"Name collision!!! {name}")
                    producers[name] = (layer, outnum)
        return producers

    def propagate_float(layer, outnum):
        nonlocal fixed_layers, fixed_outputs
        # checking own output here type for cases like float comparison
        my_hard_float = (
            layer.output_type_is_set(outnum)
            and layer.get_output_type(outnum) == trt.DataType.FLOAT
        )

        if layer.type == trt.LayerType.CONSTANT or layer.name in processed:
            return my_hard_float

        # traverse depth first
        upstream_float = False
        for innum in range(layer.num_inputs):
            inp = layer.get_input(innum)
            if inp is not None and inp.name in producers:
                producer = producers[inp.name]
                upstream_float = propagate_float(*producer) or upstream_float

        # set layer precision if any of operands are hard floats
        # and layer is float op
        need_float = upstream_float and (
            layer.precision == trt.DataType.FLOAT
            or layer.precision == trt.DataType.HALF
        )
        if need_float and not layer.output_type_is_set(outnum):
            layer.precision = trt.DataType.FLOAT
            logger.verbose(f"Marking {layer.name} as FLOAT precision")
            fixed_layers += 1
            # adjust this layer's output precision if not set yet
            if not my_hard_float:
                otype = layer.get_output_type(outnum)
                if otype == trt.DataType.FLOAT or otype == trt.DataType.HALF:
                    my_hard_float = True
                    logger.verbose(
                        f"Marking {layer.name}:{outnum} output as FLOAT"
                    )
                    layer.set_output_type(
                        index=outnum, dtype=trt.DataType.FLOAT
                    )
                    fixed_outputs += 1
        processed[layer.name] = True
        return my_hard_float

    producers = get_producers(network_definition)
    logger.info(
        f"Mixed-precision net: {len(network_definition)} layers, "
        f"{len(producers)} tensors, "
        f"{network_definition.num_outputs} outputs..."
    )
    sys.setrecursionlimit(len(network_definition))
    for numout in range(network_definition.num_outputs):
        out = network_definition.get_output(numout)
        out_layer = producers[out.name]
        propagate_float(*out_layer)
    logger.info(
        "Mixed-precision net: "
        f"{fixed_layers} layers / {fixed_outputs} outputs fixed"
    )
    return network_definition


def create_config(args, network):
    config_args = {
        "tf32": (not args.noTF32),
        "fp16": (args.precision == PRECISION_NAME_FP16),
        # NOTE: after 8.4, this option should be replaced with
        # precision_constraints.
        #"obey_precision_constraints": True,
        "precision_constraints": "obey",
    }

    # NOTE: A given workspace size is MiB,
    # but TensorRT expects this size is bytes.
    workspace_size_in_bytes = args.workspace_size * (1 << 20)
    if hasattr(trt.IBuilderConfig, "set_memory_pool_limit"):
        # Newer version after 8.4.
        config_args["memory_pool_limits"] = {
            trt.MemoryPoolType.WORKSPACE: workspace_size_in_bytes,
        }
    else:
        # Older version before 8.2.
        config_args["max_workspace_size"] = workspace_size_in_bytes

    if args.batchsize_range is not None and args.shape_ranges is not None:
        # NOTE: This script assumes only Conformer model optimization.
        assert network.num_inputs == 2

        audio_signal_input = network.get_input(0)
        length_input = network.get_input(1)

        original_input_shape = audio_signal_input.shape
        batchsize_range = list(map(int, args.batchsize_range.split(",")))
        seqlen_ranges = [
            list(map(int, s.split(","))) for s in args.shape_ranges
        ]

        assert len(original_input_shape) == 3
        assert len(batchsize_range) > 0
        if args.use_variable_batchsize:
            assert len(batchsize_range) == 3
        assert all(len(sr) == 3 for sr in seqlen_ranges)

        profiles = []
        if args.use_variable_batchsize:
            for sr in seqlen_ranges:
                profile = Profile()
                profile.add(
                    audio_signal_input.name,
                    min=(batchsize_range[0], original_input_shape[1], sr[0]),
                    opt=(batchsize_range[1], original_input_shape[1], sr[1]),
                    max=(batchsize_range[2], original_input_shape[1], sr[2]),
                )
                profile.add(
                    length_input.name,
                    min=(batchsize_range[0],),
                    opt=(batchsize_range[1],),
                    max=(batchsize_range[2],),
                )
                profiles.append(profile)
        else:
            for bs in batchsize_range:
                for sr in seqlen_ranges:
                    profile = Profile()
                    profile.add(
                        audio_signal_input.name,
                        min=(bs, original_input_shape[1], sr[0]),
                        opt=(bs, original_input_shape[1], sr[1]),
                        max=(bs, original_input_shape[1], sr[2]),
                    )
                    profile.add(
                        length_input.name,
                        min=(bs,),
                        opt=(bs,),
                        max=(bs,),
                    )
                    profiles.append(profile)
        config_args["profiles"] = profiles

    return CreateConfig(**config_args)


def verify_args(args, builder):
    if (
        args.precision == PRECISION_NAME_FP16
        and not builder.platform_has_fast_fp16
    ):
        print_with_datetime(
            "Your GPU doesn't support hardware FP16. "
            "It may cause slower infernece."
        )
    if (
        args.precision == PRECISION_NAME_INT8
        and not builder.platform_has_fast_int8
    ):
        print_with_datetime(
            "Your GPU doesn't support hardware INT8. "
            "It may cause slower infernece."
        )

    if args.noTF32:
        if not builder.platform_has_tf32:
            print_with_datetime(
                "Your GPU doesn't support TF32. " "--noTF32 will be ignored."
            )
        else:
            print_with_datetime(
                "Disabling TF32 even though your GPU has TF32 hardware. "
                "It may cause slower inference."
            )

    if (args.batchsize_range is None) ^ (args.shape_ranges is None):
        raise RuntimeError(
            "Unexpected parameters combination of "
            "batchsize_range and shape_ranges: "
            f"batchsize_range={args.batchsize_range}, "
            f"shape_ranges={args.shape_ranges}. "
            "Specifying both or not specifying neither is only acceptable."
        )


PRECISION_NAME_FP32 = "fp32"
PRECISION_NAME_FP16 = "fp16"
PRECISION_NAME_INT8 = "int8"
SUPPORTED_PRECISIONS = (
    PRECISION_NAME_FP32,
    PRECISION_NAME_FP16,
    # NOTE: For int8 support,
    # calibration steps must be implemented.
    # PRECISION_NAME_INT8,
)


def main():
    parser = argparse.ArgumentParser()

    # Basic file path information.
    parser.add_argument(
        "--onnxpath",
        required=True,
        type=str,
        help="Path to source ONNX file.",
    )
    parser.add_argument(
        "--trtengine-path",
        required=True,
        type=str,
        help="Path to save result TensorRT engine.",
    )

    # Optimization options.
    parser.add_argument(
        "--precision",
        default=PRECISION_NAME_FP32,
        choices=SUPPORTED_PRECISIONS,
        type=str,
        help="Target precision.",
    )
    parser.add_argument(
        "--noTF32",
        action="store_true",
        help="Disable TF32.",
    )
    # NOTE: Thie feature has not been implemented yet.
    # parser.add_argument(
    #     "--use-sparsity",
    #     action="store_true",
    #     help="Enable Structured Sparcity.",
    # )
    parser.add_argument(
        "--workspace-size",
        default=30000,
        type=int,
        help="Workspace size (in MiB)",
    )

    parser.add_argument(
        "--use-variable-batchsize",
        action="store_true",
        help=(
            "Enable variable batchsize. "
            "This option sets different values "
            "as a first element of each profile."
        ),
    )
    parser.add_argument(
        "--batchsize-range",
        default=None,
        type=str,
        help=(
            "Dynamic batch size candidates. "
            "Comma separated values must be specified. "
            "Each value represents each target batch size. "
            "(e.g., --batchsize-range 1,8,16 means that "
            "minimum batch size is 1, intermediate batch size is 8, "
            "and maximum batch size is 16.)"
        ),
    )
    parser.add_argument(
        "--shape-ranges",
        default=None,
        type=str,
        nargs="+",
        help=(
            "Dynamic axes shape range. "
            "At least one arg must be specified and "
            "each arg must include three int values comma separated. "
            "(e.g., --shapes-range 1,16,32 33,101,301 means that "
            "there are two optimization configurations. "
            "First config reprensents min len of sequence is 1, "
            "opt len is 16 and max len is 32. "
            "Similarly, second config represents min len is 33, "
            "opt len is 101, and max len is 301.)"
        ),
    )

    parser.add_argument(
        "--use-internal-fix",
        action="store_true",
        help="Switch a network fix method to NV internal version.",
    )

    # Misc options.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose log.",
    )
    args = parser.parse_args()
    print_with_datetime(args)

    verbosity = PG_LOGGER.EXTRA_VERBOSE if args.verbose else PG_LOGGER.WARNING
    with PG_LOGGER.verbosity(verbosity):
        print_with_datetime("Try to parse ONNX.")
        builder, network, parser = network_from_onnx_path(args.onnxpath)

        print_with_datetime("Verify given args.")
        verify_args(args, builder)
        print_with_datetime("Setup optimization configuration.")
        trt_config = create_config(args, network)

        print_with_datetime(
            "Apply some fixes and tricks to "
            "keep Conformer model accuracy in FP16."
        )
        if args.precision != PRECISION_NAME_FP32:
            if args.use_internal_fix:
                network = fix_fp16_network(network, PG_LOGGER)
            else:
                network = add_extra_fp32(network)
        print_with_datetime("Run optimization.")
        trt_engine = engine_from_network(
            (builder, network, parser),
            config=trt_config,
        )
        print_with_datetime("Save model.")
        save_engine(trt_engine, args.trtengine_path)
    print_with_datetime("done.")


if __name__ == "__main__":
    main()
