#!/usr/bin/env python3
"""
g1_thermal_logger_sdk.py
Logger PASIVO y continuo del G1 para el modelo energía-temperatura, con el SDK de Unitree
(unitree_sdk2py, DDS directo), SIN ROS 2. Mismo formato de salida que g1_thermal_logger.py (la versión ROS 2),
así que g1_logger_analysis.py lo lee igual.

Solo se suscribe (rt/lowstate, rt/lf/bmsstate, rt/lf/mainboardstate); NUNCA publica comandos, así que
puede correr mientras otras personas usan el robot.

Ciclo de vida:
  - espera hasta que llegue LowState (robot encendido)  -> abre una sesión
  - graba todo continuamente (CSV en trozos de --chunk-min minutos)
  - si LowState deja de llegar --timeout-s segundos (robot apagado / cambio de
    batería / cable desconectado) -> cierra la sesión y vuelve a esperar
  - si la batería sube de golpe (SOC +--soc-jump %) -> cierra y abre otra
    sesión (batería cambiada sin que se cortara el logger)
  - si el tick del robot se reinicia -> cierra y abre otra sesión
  - Ctrl+C cierra limpio

Salida:
  <out>/session_YYYYmmdd_HHMMSS/part_000.csv(.gz), part_001..., meta.json

Columnas por motor (29 del G1, nombres de la tabla de abajo):
  q_meas_<j>, dq_meas_<j>, tau_est_<j>, temp_<j> (temperature[0], carcasa), temp2_<j> (temperature[1],
  bobinado), motorstate_<j> (bits de error: si un motor entra en protección térmica se ve acá)
Además: t_wall, t_mono, tick, mode_machine, IMU, BMS (soc, corriente, voltajes,
temperaturas) y la placa principal en crudo (mbs_value_0..5 = voltaje_0, voltaje_1, corriente, 0, 0, 0;
mbs_temp_0..5).

Uso (en el PC conectado al robot, con unitree_sdk2py instalado; sin source de ROS):
  python3 g1_thermal_logger_sdk.py <interfaz>              # p. ej. enp3s0: la del cable al robot (ip a)
  python3 g1_thermal_logger_sdk.py <interfaz> --out ~/g1_logs --rate 50

  # para dejarlo corriendo aunque se cierre la terminal:
  nohup python3 g1_thermal_logger_sdk.py <interfaz> --out ~/g1_logs > ~/g1_logs/logger.out 2>&1 &
"""

import argparse
import gzip
import json
import os
import shutil
import signal
import threading
import time
from datetime import datetime

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, LowState_, MainBoardState_

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


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
                         temp_max_arm={}, source="unitree_sdk2py (sin ROS 2)",
                         columns_note="temp_=temperature[0] (carcasa), temp2_=temperature[1] (bobinado); motorstate_ = "
                         "bits de error del motor (en el GO-M8010-6 de Unitree el error 1 = sobrecalentamiento, protección "
                         "a 90 °C que exige reiniciar el motor; en el G1 sin documentar); "
                         "mbs_value_* = MainBoardState.value[] en crudo (0, 1 = voltajes, 2 = corriente)")
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
        """Fila libre del buffer para llenar en el lugar."""
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


class ThermalLogger:
    """Los handlers del SDK (hilo de DDS) solo guardan el último mensaje; el bucle principal toma una fila a --rate
    (LowState llega a ~500 Hz) y una vez por segundo hace flush, revisa cortes y muestra el estado."""

    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.session = None
        self.low = None              # último LowState_
        self.last_low = None         # time.monotonic() del último LowState
        self.n_low = 0
        self.last_tick = None
        self.soc = None
        self.bms_row = np.full(8, np.nan)
        self.mbs_row = np.full(13, np.nan)
        self.stop = threading.Event()

        self.subs = []
        for topic, typ, fn in ((args.lowstate_topic, LowState_, self.on_low),
                               (args.bms_topic, BmsState_, self.on_bms),
                               (args.mbs_topic, MainBoardState_, self.on_mbs)):
            sub = ChannelSubscriber(topic, typ)
            sub.Init(fn, 1)          # cola de 1 en un hilo del SDK: si se atrasa, se queda con el último
            self.subs.append(sub)
        log(f"Esperando al robot en '{args.lowstate_topic}' (BMS '{args.bms_topic}', placa '{args.mbs_topic}', "
            f"interfaz {args.iface or 'auto'}), grabando a {args.rate:g} Hz en {args.out}")

    # ---------- handlers (hilo del SDK) ----------
    def on_low(self, m):
        with self.lock:
            self.low = m
            self.last_low = time.monotonic()
            self.n_low += 1

    def on_bms(self, m):
        temps = [t for t in m.temperature if t != 0] or [0]
        row = [time.time(), m.soc, m.current, m.bmsvoltage[0], m.bmsvoltage[1], m.bmsvoltage[2],
               sum(temps) / len(temps), max(temps)]
        with self.lock:
            self.bms_row[:] = row
            self.soc = int(m.soc)
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
        with self.lock:
            self.mbs_row[0] = time.time()
            self.mbs_row[1:7] = list(m.value)
            self.mbs_row[7:13] = list(m.temperature)

    # ---------- bucle principal ----------
    def sample(self):
        """Una fila con el último LowState (si llegó uno nuevo desde la fila anterior)."""
        with self.lock:
            m, n = self.low, self.n_low
            if m is None or n == self._n_last:
                return
            self._n_last = n
            if self.session is None:
                self.session = Session(self.a.out, self.a.chunk_min, self.a.gzip, "LowState detectado")
                log(f"Robot detectado -> nueva sesión {self.session.dir}")
            # tick reiniciado = el robot se reinició aunque no notamos el corte
            if self.last_tick is not None and m.tick + 1000 < self.last_tick:
                self._restart_session(f"tick reiniciado {self.last_tick}→{m.tick}")
            self.last_tick = m.tick
            r = self.session.row()
            r[0], r[1], r[2], r[3] = time.time(), time.monotonic(), m.tick, m.mode_machine
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
            self.n_rows_status += 1

    def every_second(self):
        with self.lock:
            s = self.session
            if s is None:
                return
            if self.last_low is not None and time.monotonic() - self.last_low > self.a.timeout_s:
                log(f"Sin LowState hace {self.a.timeout_s:g} s -> cierro la sesión "
                    f"({s.rows} filas, {(time.time() - s.t0) / 3600:.2f} h). Espero al robot.")
                s.close(f"sin LowState {self.a.timeout_s:g} s (apagado / batería / desconexión)")
                self.session = None
                self.last_tick = None
                self.low = None
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
                bms = "sí" if np.isfinite(self.bms_row[0]) else "NO"
                mbs = "sí" if np.isfinite(self.mbs_row[0]) else "NO"
                log(f"{(time.time() - s.t0) / 60:.0f} min, {self.n_rows_status / dt:.0f} Hz grabando, SOC {self.soc}%, "
                    f"BMS {bms}, placa {mbs}, T máx brazos: {hot}")
                self.last_status = time.monotonic()
                self.n_rows_status = 0

    def run(self):
        self._n_last = 0
        self.n_rows_status = 0
        self.last_status = time.monotonic()
        period = 1.0 / self.a.rate
        t_next = time.monotonic()
        t_sec = t_next + 1.0
        while not self.stop.is_set():
            self.sample()
            now = time.monotonic()
            if now >= t_sec:
                self.every_second()
                t_sec = now + 1.0
            t_next += period
            if t_next < now:     # atrasado (p. ej. flush lento): no recuperar de golpe
                t_next = now + period
            self.stop.wait(max(0.0, t_next - time.monotonic()))

    def _restart_session(self, reason):
        log(reason + " -> nueva sesión")
        self.session.close(reason)
        self.session = Session(self.a.out, self.a.chunk_min, self.a.gzip, reason)
        self.last_tick = None

    def shutdown(self):
        with self.lock:
            if self.session is not None:
                self.session.close("Ctrl+C / fin del logger")
                log(f"Sesión cerrada: {self.session.dir}")
                self.session = None


def main():
    ap = argparse.ArgumentParser(description="Logger pasivo continuo del G1 con unitree_sdk2py (sin ROS 2).")
    ap.add_argument("iface", nargs="?", default=None,
                    help="interfaz de red conectada al robot (p. ej. enp3s0, ver 'ip a'); sin ella, automática")
    ap.add_argument("--domain", type=int, default=0, help="dominio DDS (el G1 usa 0)")
    ap.add_argument("--out", default=os.path.expanduser("~/g1_logs"))
    ap.add_argument("--rate", type=float, default=50.0, help="Hz de grabación (LowState llega a ~500 Hz)")
    ap.add_argument("--chunk-min", type=float, default=10.0, help="minutos por archivo CSV")
    ap.add_argument("--timeout-s", type=float, default=5.0, help="segundos sin LowState para cerrar la sesión")
    ap.add_argument("--soc-jump", type=int, default=5, help="subida de SOC (%%) que se toma como batería nueva")
    ap.add_argument("--status-s", type=float, default=60.0, help="cada cuánto imprimir el estado")
    ap.add_argument("--no-gzip", dest="gzip", action="store_false", help="no comprimir los trozos cerrados")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--bms-topic", default="rt/lf/bmsstate")
    ap.add_argument("--mbs-topic", default="rt/lf/mainboardstate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.iface:
        ChannelFactoryInitialize(args.domain, args.iface)
    else:
        ChannelFactoryInitialize(args.domain)
    lg = ThermalLogger(args)

    def stop(*_):
        lg.stop.set()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        lg.run()
    finally:
        lg.shutdown()


if __name__ == "__main__":
    main()