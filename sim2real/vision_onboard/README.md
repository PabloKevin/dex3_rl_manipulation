# vision_onboard — the apple detector on the Jetson's GPU

The laptop's RF-DETR apple detector (`g1_fruits_picker/models/rfdetr_personal`), run as a TensorRT engine on the
robot's Orin NX, publishing the same `/fruits_detected_raw` message the laptop's `live_locator` publishes.
Everything downstream (fruits_to_torso, the picker's tracker, the toy, the dashboard) is unchanged.

**RL runs only, one detector at a time:** `./g1_launch.sh toy --vision onboard -- harvest ...` starts this instead of
the laptop's live_locator. The default (`--vision laptop`) and the original test are untouched. If the Jetson's
detector does not come up, the launcher starts the laptop's as before.

Nothing is installed on the robot: system Python 3.8, JetPack 5.1.1's TensorRT 8.5, ROS 2 Foxy (all already there).
Everything lives in `~/vision_onboard` on the Jetson; deleting that folder undoes it.

## Files
| file | what |
|---|---|
| `vision_onboard.py` | the ROS 2 node (robot): raw colour + aligned depth (local) → detections → `/fruits_detected_raw` + the annotated picture |
| `trt_rfdetr.py` | the engine with rfdetr's exact pre/post-processing (GPU buffers through libcudart, no pycuda) |
| `apple_depth.py` | the laptop's apple-depth code, copied verbatim from `common/object_locator.py` |
| `build_engine.py` | ONNX → engine: FP16, layer norms pinned to FP32 (plain FP16 gives NaN) |
| `vision_onboard_robot.sh` | start / stop / status on the robot (cores 0-5, nice 10; log `/tmp/vision_onboard.log`) |
| `deploy_vision_onboard.sh` | copy the code to the Jetson |

## Model export (laptop, once per new checkpoint)
`rfdetr` export at **opset 16** (TensorRT 8.5 cannot read opset 17's LayerNormalization) and **560×560** (other
sizes need an op the exporter lacks). The `onnx` package only for the export, outside the live environment:

    pip install --target /tmp/onnx_pkgs "onnx<1.18"   # then remove the numpy it pulls in
    PYTHONPATH=/tmp/onnx_pkgs python -c "from rfdetr import RFDETRBase; m = RFDETRBase(pretrain_weights='.../checkpoint_best_total.pth', num_classes=3, device='cpu'); m.export(output_dir='out', opset_version=16, shape=(560, 560))"

Then **simplify** it (onnxsim, also outside the live environment). Without this, TensorRT 8.5 builds an engine that
outputs NaN for every box, in FP16 *and* FP32 (2026-10-21); simplified (1978 -> 1172 nodes, no Where / Shape /
ConstantOfShape left) it matches the laptop: worst box IoU 0.992, scores within 0.015, edges within 0.1 px (6 frames):

    PYTHONPATH=/tmp/sim_pkgs python -c "import onnx, onnxsim; m, ok = onnxsim.simplify(onnx.load('out/inference_model.onnx')); onnx.save(m, 'rfdetr560_op16_sim.onnx')"

Then on the Jetson: `python3 build_engine.py models/rfdetr560_op16_sim.onnx models/rfdetr560_mixed.engine`
(about 10 min). Speed: 28 ms in the engine, 51 ms per frame in all (pre/post-processing on the CPU).

## Not covered (yet)
- The toy's camera recordings (run folders) come from live_locator's frame recorder: not recorded with `--vision onboard`.
- The head camera runs at 6 frames/s (`~/ros2_cameras/multi_camera.launch.py`, shared with the original test):
  detection on the Jetson does not raise that.
