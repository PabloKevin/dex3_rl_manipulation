#!/usr/bin/env python3
"""trt_rfdetr.py -- the laptop's RF-DETR apple detector (g1_fruits_picker/models/rfdetr_personal), as a TensorRT FP16
engine on the Jetson's GPU. Same pre/post-processing as rfdetr.RFDETRBase.predict (what live_locator runs):
RGB / 255 -> resize to 560x560 (bilinear) -> ImageNet mean/std -> engine -> sigmoid, top 300 of (query, class),
class = index % 4, boxes cxcywh (normalised) -> xyxy pixels; kept when score > threshold.

Runs on the Jetson's system Python 3.8 with JetPack's TensorRT 8.5 -- nothing installed: the GPU buffers go through
libcudart with ctypes (no pycuda). Engine: built once with trtexec (see README in this folder).
"""
from __future__ import annotations

import ctypes
import time

import cv2
import numpy as np

if not hasattr(np, "bool"):        # TensorRT 8.5's bindings still use np.bool (gone in the Jetson's numpy 1.24)
    np.bool = bool
import tensorrt as trt  # noqa: E402

MEANS = np.array([0.485, 0.456, 0.406], np.float32)
STDS = np.array([0.229, 0.224, 0.225], np.float32)
CLASS_NAMES = ["red_apple", "green_apple", "orange"]

_cudart = ctypes.CDLL("libcudart.so")
_H2D, _D2H = 1, 2
# 64-bit pointers: without these, ctypes passes Python ints as 32-bit C ints (truncated device addresses -> garbage)
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
for _f in ("cudaMalloc", "cudaMemcpyAsync", "cudaStreamCreate", "cudaStreamSynchronize"):
    getattr(_cudart, _f).restype = ctypes.c_int


def _ck(err, what):
    if err != 0:
        raise RuntimeError(f"CUDA {what} failed: error {err}")


class Detector:
    def __init__(self, engine_path: str, threshold: float = 0.3, num_select: int = 300):
        self.threshold, self.num_select = float(threshold), int(num_select)
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.bufs, self.host, self.dev = [], {}, {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            h = np.zeros(shape, dtype=dtype)
            d = ctypes.c_void_p()
            _ck(_cudart.cudaMalloc(ctypes.byref(d), ctypes.c_size_t(h.nbytes)), "cudaMalloc")
            self.host[name], self.dev[name] = h, d
            self.bufs.append(d.value)
        self.in_name = self.engine.get_binding_name(0)
        _, _, self.H, self.W = self.host[self.in_name].shape
        self.stream = ctypes.c_void_p()
        _ck(_cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")

    def _copy(self, dst, src, n, kind):
        _ck(_cudart.cudaMemcpyAsync(dst, src, n, kind, self.stream.value), "cudaMemcpyAsync")

    def infer_raw(self, rgb: np.ndarray):
        """rgb uint8 HxWx3 -> (boxes cxcywh normalised [300, 4], logits [300, K])."""
        x = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - MEANS) / STDS).transpose(2, 0, 1)[None]
        inp = self.host[self.in_name]
        inp[...] = x.astype(inp.dtype)
        self._copy(self.dev[self.in_name].value, inp.ctypes.data, inp.nbytes, _H2D)
        if not self.ctx.execute_async_v2(self.bufs, self.stream.value):
            raise RuntimeError("TensorRT execute failed")
        for name, h in self.host.items():
            if name != self.in_name:
                self._copy(h.ctypes.data, self.dev[name].value, h.nbytes, _D2H)
        _ck(_cudart.cudaStreamSynchronize(self.stream.value), "cudaStreamSynchronize")
        return self.host["dets"][0].astype(np.float32), self.host["labels"][0].astype(np.float32)

    def detect(self, rgb: np.ndarray):
        """rgb uint8 HxWx3 -> [(x1, y1, x2, y2, score, class_id, class_label)] in pixels of rgb."""
        h, w = rgb.shape[:2]
        boxes, logits = self.infer_raw(rgb)
        prob = 1.0 / (1.0 + np.exp(-logits))
        K = prob.shape[1]
        flat = prob.ravel()
        top = np.argpartition(-flat, self.num_select - 1)[:self.num_select]
        top = top[np.argsort(-flat[top])]
        out = []
        for idx in top:
            s = float(flat[idx])
            if s <= self.threshold:
                break
            q, c = divmod(int(idx), K)
            cx, cy, bw, bh = boxes[q]
            x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            out.append((float(x1), float(y1), float(x2), float(y2), s, c,
                        CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)))
        return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="models/rfdetr560_fp16.engine")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--bench", type=int, default=0)
    a = ap.parse_args()
    det = Detector(a.engine)
    res = {}
    for p in a.images:
        bgr = cv2.imread(p)
        res[p] = det.detect(bgr[:, :, ::-1].copy())
    print(json.dumps(res))
    if a.bench:
        rgb = np.zeros((240, 424, 3), np.uint8)
        t0 = time.perf_counter()
        for _ in range(a.bench):
            det.detect(rgb)
        print(f"bench: {1000 * (time.perf_counter() - t0) / a.bench:.1f} ms per frame (pre + engine + post)")
