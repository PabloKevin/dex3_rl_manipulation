#!/usr/bin/env python3
"""
g1_thermal_logger.py
Logger PASIVO y continuo del G1 para el modelo energía-temperatura.

Solo se suscribe (lowstate, BMS, placa principal); NUNCA publica comandos, así
que puede correr mientras otras personas usan el robot.

Ciclo de vida:
  - espera hasta que llegue LowState (robot encendido)  -> abre una sesión
  - graba todo continuamente (CSV en trozos de --chunk-min minutos)
  - si LowState deja de llegar --timeout-s segundos (robot apagado / cambio de
    batería / cable desconectado) -> cierra la sesión y vuelve a esperar
  - si la batería sube de golpe (SOC +--soc-jump %) -> cierra y abre otra
    sesión (batería cambiada sin que se cortara el logger)
  - Ctrl+C cierra limpio

Salida:
  <out>/session_YYYYmmdd_HHMMSS/part_000.csv(.gz), part_001..., meta.json

Columnas por motor (29 del G1, nombres de la tabla de abajo):
  q_meas_<j>, dq_meas_<j>, tau_est_<j>, temp_<j> (temperature[0]), temp2_<j> (temperature[1]),
  motorstate_<j> (bits de error: si un motor entra en protección térmica se ve acá)
Además: t_wall, t_mono, tick, mode_machine, IMU, BMS (soc, corriente, voltajes,
temperaturas) y la placa principal en crudo (mbs_value_0..5, mbs_temp_0..5).

Uso (en el PC conectado al robot):
  source /opt/ros/humble/setup.bash
  source ~/ros2_ws/install/setup.bash      # para unitree_hg
  python3 g1_thermal_logger.py --out ~/g1_logs

  # para dejarlo corriendo aunque se cierre la terminal:
  nohup python3 g1_thermal_logger.py --out ~/g1_logs > ~/g1_logs/logger.out 2>&1 &
"""

import argparse
import gzip
import json
import os
import shutil
import struct
import signal
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message

from unitree_hg.msg import LowState, BmsState, MainBoardState


# Orden de motor_state[] en el G1 de 29 DOF (los índices 29..34 no se usan).
G1_JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
ARM_IDX = list(range(15, 29))   # para el resumen en pantalla


def columns():
    cols = ["t_wall", "t_mono", "tick", "mode_machine"]
    for j in G1_JOINTS:
        cols += [f"q_meas_{j}", f"dq_meas_{j}", f"tau_est_{j}", f"temp_{j}", f"temp2_{j}", f"motorstate_{j}"]
    cols += ["imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z",
             "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
             "imu_acc_x", "imu_acc_y", "imu_acc_z",
             "imu_roll", "imu_pitch", "imu_yaw", "imu_temp"]
    cols += ["bms_t", "bms_soc", "bms_current", "bms_voltage_0", "bms_voltage_1", "bms_voltage_2",
             "bms_temp_avg", "bms_temp_max"]
    cols += ["mbs_t"] + [f"mbs_value_{i}" for i in range(6)] + [f"mbs_temp_{i}" for i in range(6)]
    return cols


N_COLS = len(columns())
# formato por columna para np.savetxt (t_wall/t_mono con décimas de ms, tick y modo enteros, el resto 5 cifras)
FMT = ["%.4f", "%.4f", "%d", "%d"] + ["%.5g"] * (N_COLS - 4)
for _i, _c in enumerate(columns()):
    if _c.startswith("motorstate_"):
        FMT[_i] = "%d"
for _c in ("bms_t", "mbs_t"):  # horas de la última lectura de batería / placa
    FMT[columns().index(_c)] = "%.2f"
# LowState llega en CDR: 4 bytes de encapsulado + version uint32[2] + mode_pr, mode_machine uint8 + 2 de
# relleno -> tick (uint32) en el byte 16. Se lee de cada mensaje sin deserializar (para ver reinicios).
TICK_OFF = 16


class Session:
    """Una sesión = desde que aparece el robot hasta que desaparece (o cambia la batería)."""

    def __init__(self, out_dir, chunk_min, use_gzip, reason):
        self.t0 = time.time()
        self.dir = os.path.join(out_dir, "session_" + datetime.fromtimestamp(self.t0).strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.chunk_s = chunk_min * 60.0
        self.use_gzip = use_gzip
        self.part = -1
        self.fh = None
        self.part_t0 = 0.0
        self.rows = 0
        self.buf = np.empty((512, N_COLS))   # filas que esperan al próximo flush (1 por segundo)
        self.n_buf = 0
        self.meta = dict(start=datetime.fromtimestamp(self.t0).isoformat(), start_reason=reason,
                         end=None, end_reason=None, rows=0, parts=[], soc_start=None, soc_end=None,
                         temp_max_arm={}, columns_note="temp_=temperature[0], temp2_=temperature[1]; motorstate_ = "
                         "bits de error del motor (en el GO-M8010-6 de Unitree el error 1 = sobrecalentamiento, protección "
                         "a 90 °C que exige reiniciar el motor; en el G1 sin documentar); "
                         "mbs_value_* = MainBoardState.value[] en crudo")
        self._new_part()

    def _new_part(self):
        self._close_part()
        self.part += 1
        name = f"part_{self.part:03d}.csv"
        self.path = os.path.join(self.dir, name)
        self.fh = open(self.path, "w", buffering=1 << 16)
        self.fh.write(",".join(columns()) + "\n")
        self.part_t0 = time.time()
        self.meta["parts"].append(name)

    def _write_buf(self):
        if self.n_buf and self.fh is not None:
            np.savetxt(self.fh, self.buf[:self.n_buf], fmt=FMT, delimiter=",")
        self.n_buf = 0

    def _close_part(self):
        if self.fh is None:
            return
        self._write_buf()
        self.fh.close()
        path = self.path
        self.fh = None
        if self.use_gzip:
            def gz():
                with open(path, "rb") as fi, gzip.open(path + ".gz", "wb", compresslevel=3) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(path)
            threading.Thread(target=gz, daemon=False).start()

    def row(self):
        """Fila libre del buffer para llenar en el lugar (sin armar listas ni strings por mensaje)."""
        if self.n_buf == len(self.buf):
            self._write_buf()
        r = self.buf[self.n_buf]
        self.n_buf += 1
        self.rows += 1
        return r

    def flush(self):
        if time.time() - self.part_t0 >= self.chunk_s:
            self._new_part()
        if self.fh:
            self._write_buf()
            self.fh.flush()
            os.fsync(self.fh.fileno())
        self.save_meta()

    def save_meta(self):
        self.meta["rows"] = self.rows
        self.meta["duration_h"] = round((time.time() - self.t0) / 3600, 3)
        tmp = os.path.join(self.dir, "meta.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "meta.json"))

    def close(self, reason):
        self.meta["end"] = datetime.now().isoformat()
        self.meta["end_reason"] = reason
        self._close_part()
        self.save_meta()


class ThermalLogger(Node):
    def __init__(self, args):
        super().__init__("g1_thermal_logger")
        self.a = args
        self.period = 1.0 / args.rate
        self.lock = threading.Lock()
        self.session = None
        self.last_low = None        # time.monotonic() del último LowState
        self.last_write = 0.0
        self.last_tick = None
        self.soc = None
        self.last_status = time.monotonic()
        self.n_since_status = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        # LowState llega a ~500 Hz: recibirlo con un callback despertaba a Python 500 veces por segundo
        # (rclpy además convertía cada mensaje de 35 motores a objetos). Ahora la suscripción vive en un
        # nodo que NO entra al executor, con cola de 1 (DDS, en C, descarta los viejos) y un timer a --rate
        # toma solo el último, en bruto; se deserializa solo lo que se graba.
        self.low_node = rclpy.create_node("g1_thermal_logger_lowstate")
        qos1 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.low_sub = self.low_node.create_subscription(LowState, args.lowstate_topic, lambda _m: None, qos1, raw=True)
        self.create_timer(self.period, self.poll_low)
        self.bms_row = np.full(8, np.nan)
        self.mbs_row = np.full(13, np.nan)
        self.create_subscription(BmsState, args.bms_topic, self.on_bms, qos)
        self.create_subscription(MainBoardState, args.mbs_topic, self.on_mbs, qos)
        self.create_timer(1.0, self.on_timer)
        self.get_logger().info(
            f"Esperando al robot en '{args.lowstate_topic}' (BMS '{args.bms_topic}', placa '{args.mbs_topic}'), "
            f"grabando a {args.rate:g} Hz en {args.out}")

    # ---------- callbacks ----------
    def on_bms(self, m):
        temps = [t for t in m.temperature if t != 0] or [0]
        self.soc = int(m.soc)
        self.bms_row[:] = [time.time(), m.soc, m.current, m.bmsvoltage[0], m.bmsvoltage[1], m.bmsvoltage[2],
                           sum(temps) / len(temps), max(temps)]
        with self.lock:
            s = self.session
            if s is None:
                return
            if s.meta["soc_start"] is None:
                s.meta["soc_start"] = self.soc
            # batería cambiada sin que se cortara LowState: SOC sube de golpe
            if s.meta["soc_end"] is not None and self.soc - s.meta["soc_end"] >= self.a.soc_jump:
                self._restart_session(f"SOC saltó {s.meta['soc_end']}→{self.soc} % (batería nueva)")
                return
            s.meta["soc_end"] = self.soc

    def on_mbs(self, m):
        self.mbs_row[0] = time.time()
        self.mbs_row[1:7] = m.value
        self.mbs_row[7:13] = m.temperature

    def poll_low(self):
        with self.low_sub.handle:
            got = self.low_sub.handle.take_message(LowState, True)
        if got is not None:
            self.on_low(got[0])

    def on_low(self, raw):
        now_m = time.monotonic()
        self.last_low = now_m
        tick = struct.unpack_from("<I", raw, TICK_OFF)[0]
        with self.lock:
            if self.session is None:
                self.session = Session(self.a.out, self.a.chunk_min, self.a.gzip, "LowState detectado")
                self.get_logger().info(f"Robot detectado -> nueva sesión {self.session.dir}")
            # tick reiniciado = el robot se reinició aunque no notamos el corte
            if self.last_tick is not None and tick + 1000 < self.last_tick:
                self._restart_session(f"tick reiniciado {self.last_tick}→{tick}")
            self.last_tick = tick
            if now_m - self.last_write < 0.5 * self.period:
                return
            self.last_write = now_m
            m = deserialize_message(raw, LowState)
            r = self.session.row()
            r[0], r[1], r[2], r[3] = time.time(), now_m, m.tick, m.mode_machine
            ms = m.motor_state
            c = 4
            for i in range(len(G1_JOINTS)):
                s = ms[i]
                t = s.temperature
                r[c:c + 6] = (s.q, s.dq, s.tau_est, t[0], t[1], s.motorstate)
                c += 6
            imu = m.imu_state
            r[c:c + 4] = imu.quaternion
            r[c + 4:c + 7] = imu.gyroscope
            r[c + 7:c + 10] = imu.accelerometer
            r[c + 10:c + 13] = imu.rpy
            r[c + 13] = imu.temperature
            c += 14
            r[c:c + 8] = self.bms_row
            r[c + 8:c + 21] = self.mbs_row
            self.n_since_status += 1

    def on_timer(self):
        with self.lock:
            s = self.session
            if s is None:
                return
            if self.last_low is not None and time.monotonic() - self.last_low > self.a.timeout_s:
                self.get_logger().warn(f"Sin LowState hace {self.a.timeout_s:g} s -> cierro la sesión "
                                       f"({s.rows} filas, {(time.time() - s.t0) / 3600:.2f} h). Espero al robot.")
                s.close(f"sin LowState {self.a.timeout_s:g} s (apagado / batería / desconexión)")
                self.session = None
                self.last_tick = None
                return
            if s.n_buf:  # T máx de los brazos (1 vez por segundo, con la última fila)
                last = s.buf[s.n_buf - 1]
                tmax = s.meta["temp_max_arm"]
                for i in ARM_IDX:
                    tj = int(last[4 + 6 * i + 3])
                    if tj > tmax.get(G1_JOINTS[i], -999):
                        tmax[G1_JOINTS[i]] = tj
            s.flush()
            if time.monotonic() - self.last_status >= self.a.status_s:
                dt = time.monotonic() - self.last_status
                tm = s.meta["temp_max_arm"]
                hot = ", ".join(f"{k.replace('left_', 'L_').replace('right_', 'R_')} {v}"
                                for k, v in sorted(tm.items(), key=lambda kv: -kv[1])[:3])
                self.get_logger().info(f"{(time.time() - s.t0) / 60:.0f} min, {self.n_since_status / dt:.0f} Hz, "
                                       f"SOC {self.soc}%, T máx brazos: {hot}")
                self.last_status = time.monotonic()
                self.n_since_status = 0

    def _restart_session(self, reason):
        self.get_logger().warn(reason + " -> nueva sesión")
        self.session.close(reason)
        self.session = Session(self.a.out, self.a.chunk_min, self.a.gzip, reason)
        self.last_tick = None

    def shutdown(self):
        with self.lock:
            if self.session is not None:
                self.session.close("Ctrl+C / fin del logger")
                self.get_logger().info(f"Sesión cerrada: {self.session.dir}")
                self.session = None


def main():
    ap = argparse.ArgumentParser(description="Logger pasivo continuo del G1 (temperatura, torque, energía).")
    ap.add_argument("--out", default=os.path.expanduser("~/g1_logs"))
    ap.add_argument("--rate", type=float, default=50.0, help="Hz de grabación (LowState llega a ~500 Hz)")
    ap.add_argument("--chunk-min", type=float, default=10.0, help="minutos por archivo CSV")
    ap.add_argument("--timeout-s", type=float, default=5.0, help="segundos sin LowState para cerrar la sesión")
    ap.add_argument("--soc-jump", type=int, default=5, help="subida de SOC (%%) que se toma como batería nueva")
    ap.add_argument("--status-s", type=float, default=60.0, help="cada cuánto imprimir el estado")
    ap.add_argument("--no-gzip", dest="gzip", action="store_false", help="no comprimir los trozos cerrados")
    ap.add_argument("--lowstate-topic", default="lowstate")
    ap.add_argument("--bms-topic", default="lf/bmsstate")
    ap.add_argument("--mbs-topic", default="lf/mainboardstate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def on_sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)   # kill / apagado del PC -> cierre limpio

    rclpy.init()
    node = ThermalLogger(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.low_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
