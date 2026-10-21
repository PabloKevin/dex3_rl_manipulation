#!/usr/bin/env python3
"""build_engine.py -- the RF-DETR ONNX (opset 16) -> a TensorRT engine on the Jetson (TensorRT 8.5).

FP16 everywhere EXCEPT the layer norms: opset 16 writes them as ReduceMean / Pow / Sqrt / Div, and in FP16 the squares
overflow -> the plain `trtexec --fp16` engine gave NaN for every box (2026-10-21). Those layers (and the softmaxes)
are pinned to FP32 with OBEY_PRECISION_CONSTRAINTS.

    python3 build_engine.py models/rfdetr560_op16.onnx models/rfdetr560_mixed.engine [--fp32]
"""
import sys

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
import tensorrt as trt  # noqa: E402


def main():
    onnx_path, out_path = sys.argv[1], sys.argv[2]
    fp32_only = "--fp32" in sys.argv
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    n32 = 0
    if not fp32_only:
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        keep = {trt.LayerType.REDUCE, trt.LayerType.SOFTMAX, trt.LayerType.UNARY}
        pow_like = {trt.ElementWiseOperation.POW, trt.ElementWiseOperation.DIV, trt.ElementWiseOperation.SUB}
        for i in range(net.num_layers):
            L = net.get_layer(i)
            pin = L.type in keep
            if L.type == trt.LayerType.ELEMENTWISE:
                L.__class__ = trt.IElementWiseLayer
                pin = L.op in pow_like
            if pin:
                L.precision = trt.float32
                for k in range(L.num_outputs):
                    L.set_output_type(k, trt.float32)
                n32 += 1
    print(f"layers {net.num_layers}, pinned to FP32: {n32}" + (" (all FP32)" if fp32_only else ""))
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        sys.exit("build failed")
    with open(out_path, "wb") as f:
        f.write(plan)
    print("saved", out_path)


if __name__ == "__main__":
    main()
