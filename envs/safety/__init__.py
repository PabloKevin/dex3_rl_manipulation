# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Runtime safety for the RL policies: the arm watchdog (monitor.py).
monitor.py imports torch only -- load it by path (importlib; register it in
sys.modules before exec_module, it holds a dataclass) where the
envs package (Isaac Lab) is not available, e.g. the sim2real runtime."""
