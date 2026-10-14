#!/usr/bin/env python3
"""
g1_thermal_logger.py
Logger PASIVO y continuo del G1 para el modelo energía-temperatura.

Solo se suscribe (lowstate, BMS, placa principal); NUNCA publica comandos, así
que puede correr mientras otras personas usan el robot.

Usa unitree_sdk2py (el mismo DDS que el picker): NO hace falta ROS 2 ni el paquete
unitree_hg. Tópicos DDS: rt/lowstate, rt/lf/bmsstate, rt/lf/mainboardstate.

Ciclo de vida:
  - espera hasta que llegue LowState (robot encendido)  -> abre una sesión
  - graba todo continuamente (CSV en trozos de --chunk-min minutos)
  - si LowState deja de llegar --timeout-s segundos (robot apagado / cambio de
    batería / cable desconectado) -> cierra la sesión y vuelve a esperar
  - si la batería sube de golpe (SOC +--soc-jump %) -> cierra y abre otra
    sesión (batería cambiada sin que se cortara el logger)
  - Ctrl+C / SIGTERM cierran limpio

Salida:
  <out>/session_YYYYmmdd_HHMMSS/part_000.csv(.gz), part_001..., meta.json
  (mismas columnas que la versión ROS 2: los análisis anteriores siguen sirviendo)

Columnas por motor (29 del G1, nombres de la tabla de abajo):
  q_meas_<j>, dq_meas_<j>, tau_est_<j>, temp_<j> (temperature[0]), temp2_<j> (temperature[1])
Además: t_wall, t_mono, tick, mode_machine, IMU, BMS (soc, corriente, voltajes,
temperaturas) y la placa principal en crudo (mbs_value_0..5, mbs_temp_0..5).

CPU (el logger comparte el portátil con el lazo de control de 50 Hz):
  - LowState llega a ~500 Hz y se graba a --rate (50 Hz). Los lectores tienen historia KEEP_LAST 1
    y se sondean a --rate: las muestras superadas se sobrescriben en C y NUNCA se deserializan en
    Python (deserializar 35 motores por mensaje es lo caro: medido ~37 % de un núcleo si se
    atienden los 500 Hz con un listener, ver abajo). Sin listener ni hilos de DDS propios.
  - --nice (10 por defecto) baja la prioridad del proceso; --cpu fija un núcleo.
  - los trozos cerrados se comprimen en un `nice gzip` aparte (no en este proceso).

Uso (en el PC conectado al robot, mismo entorno que el picker):
  conda activate g1_dex3_sim2real
  python3 sim2real/g1_thermal_logger.py --out ~/g1_logs            # interfaz por defecto: enp62s0

  # para dejarlo corriendo aunque se cierre la terminal:
  nohup python3 sim2real/g1_thermal_logger.py --out ~/g1_logs > ~/g1_logs/logger.out 2>&1 &
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime

from cyclonedds.core import DDSException
from cyclonedds.internal import InvalidSample
from cyclonedds.qos import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
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
        cols += [f"q_meas_{j}", f"dq_meas_{j}", f"tau_est_{j}", f"temp_{j}", f"temp2_{j}"]
    cols += ["imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z",
             "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
             "imu_acc_x", "imu_acc_y", "imu_acc_z",
             "imu_roll", "imu_pitch", "imu_yaw", "imu_temp"]
    cols += ["bms_t", "bms_soc", "bms_current", "bms_voltage_0", "bms_voltage_1", "bms_voltage_2",
             "bms_temp_avg", "bms_temp_max"]
    cols += ["mbs_t"] + [f"mbs_value_{i}" for i in range(6)] + [f"mbs_temp_{i}" for i in range(6)]
    return cols


def f4(x):
    return f"{x:.5g}"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class Session:
    """Una sesión = desde que aparece el robot hasta que desaparece (o cambia la batería)."""

    def __init__(self, out_dir, chunk_min, use_gzip, reason):
        self.t0 = time.time()
        self.dir = os.path.join(out_dir, "session_" + datetime.fromtimestamp(self.t0).strftime("%Y%m%d_%H%M%S"))
        n = 1
        while os.path.exists(self.dir):         # dos sesiones en el mismo segundo no se pisan
            n += 1
            self.dir = os.path.join(out_dir, "session_" + datetime.fromtimestamp(self.t0).strftime("%Y%m%d_%H%M%S")
                                    + f"_{n}")
        os.makedirs(self.dir)
        self.chunk_s = chunk_min * 60.0
        self.use_gzip = use_gzip
        self.part = -1
        self.fh = None
        self.writer = None
        self.part_t0 = 0.0
        self.rows = 0
        self.gz = []                            # procesos gzip en curso
        self.meta = dict(start=datetime.fromtimestamp(self.t0).isoformat(), start_reason=reason,
                         end=None, end_reason=None, rows=0, parts=[], soc_start=None, soc_end=None,
                         temp_max_arm={}, columns_note="temp_=temperature[0], temp2_=temperature[1]; "
                         "mbs_value_* = MainBoardState.value[] en crudo")
        self._new_part()

    def _new_part(self):
        self._close_part()
        self.part += 1
        name = f"part_{self.part:03d}.csv"
        self.path = os.path.join(self.dir, name)
        self.fh = open(self.path, "w", newline="", buffering=1 << 16)
        self.writer = csv.writer(self.fh)
        self.writer.writerow(columns())
        self.part_t0 = time.time()
        self.meta["parts"].append(name)

    def _close_part(self):
        if self.fh is None:
            return
        self.fh.close()
        path = self.path
        self.fh = None
        if self.use_gzip:
            # fuera de este proceso y con prioridad baja: gzip reemplaza part_N.csv por part_N.csv.gz
            try:
                self.gz.append(subprocess.Popen(["nice", "-n", "15", "gzip", "-f", path],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            except OSError as e:
                log(f"no pude lanzar gzip ({e}): {os.path.basename(path)} queda sin comprimir")
        self.gz = [p for p in self.gz if p.poll() is None]

    def write(self, row):
        if time.time() - self.part_t0 >= self.chunk_s:
            self._new_part()
        self.writer.writerow(row)
        self.rows += 1

    def flush(self):
        if self.fh:
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
        for p in self.gz:                       # que no queden .csv sin comprimir al salir
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass


class ThermalLogger:
    def __init__(self, args):
        self.a = args
        self.period = 1.0 / args.rate
        self.lock = threading.Lock()
        self.session = None
        self.last_low = None        # time.monotonic() del último LowState
        self.last_write = 0.0
        self.last_tick = None
        self.bms = None             # (t_wall, valores)
        self.mbs = None
        self.soc = None
        self.last_status = time.monotonic()
        self.n_since_status = 0
        self.readers = []

    def start(self):
        a = self.a
        ChannelFactoryInitialize(a.domain, a.net_if)
        # el participante que el SDK ya creó (mismo dominio / interfaz que el resto del picker)
        participant = getattr(ChannelFactory, "_ChannelFactory__participant", None)
        if participant is None:
            raise RuntimeError("unitree_sdk2py cambió: no encuentro el participante DDS de ChannelFactory")
        # BEST_EFFORT + KEEP_LAST 1: compatible con cualquier escritor y sin cola que deserializar
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self.readers = []
        for name, typ, handler in ((a.lowstate_topic, LowState_, self.on_low),
                                   (a.bms_topic, BmsState_, self.on_bms),
                                   (a.mbs_topic, MainBoardState_, self.on_mbs)):
            self.readers.append((DataReader(participant, Topic(participant, name, typ), qos), handler))
        log(f"Esperando al robot en '{a.lowstate_topic}' (BMS '{a.bms_topic}', placa '{a.mbs_topic}'), "
            f"dominio {a.domain}, interfaz {a.net_if or 'auto'}, grabando a {a.rate:g} Hz en {a.out}")

    def poll(self):
        """Una muestra (la última) de cada lector; las anteriores ya fueron sobrescritas."""
        for reader, handler in self.readers:
            try:
                got = reader.take(1)
            except DDSException as e:
                log(f"lector DDS: {e.msg}")
                continue
            for m in got:
                if not isinstance(m, InvalidSample):
                    handler(m)

    # ---------- manejadores (los llama poll(), en el hilo principal) ----------
    def on_bms(self, m):
        temps = [t for t in m.temperature if t != 0] or [0]
        self.soc = int(m.soc)
        self.bms = (time.time(), [int(m.soc), int(m.current), int(m.bmsvoltage[0]), int(m.bmsvoltage[1]),
                                  int(m.bmsvoltage[2]), sum(temps) / len(temps), max(temps)])
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
        self.mbs = (time.time(), [float(v) for v in m.value] + [int(t) for t in m.temperature])

    def on_low(self, m):
        now_m = time.monotonic()
        self.last_low = now_m
        with self.lock:
            if self.session is None:
                self.session = Session(self.a.out, self.a.chunk_min, self.a.gzip, "LowState detectado")
                log(f"Robot detectado -> nueva sesión {self.session.dir}")
            # tick reiniciado = el robot se reinició aunque no notamos el corte
            if self.last_tick is not None and m.tick + 1000 < self.last_tick:
                self._restart_session(f"tick reiniciado {self.last_tick}→{m.tick}")
            self.last_tick = m.tick
            # se sondea a 1/rate, así que casi siempre toca; 0.8: la fluctuación del sondeo no debe
            # tirar una muestra buena
            if now_m - self.last_write < 0.8 * self.period:
                return
            self.last_write = now_m
            row = [f"{time.time():.4f}", f"{now_m:.4f}", m.tick, m.mode_machine]
            ms = m.motor_state
            tmax = self.session.meta["temp_max_arm"]
            for i, j in enumerate(G1_JOINTS):
                s = ms[i]
                row += [f4(s.q), f4(s.dq), f4(s.tau_est), int(s.temperature[0]), int(s.temperature[1])]
                if i in ARM_IDX and s.temperature[0] > tmax.get(j, -999):
                    tmax[j] = int(s.temperature[0])
            imu = m.imu_state
            row += [f4(v) for v in imu.quaternion] + [f4(v) for v in imu.gyroscope] + \
                   [f4(v) for v in imu.accelerometer] + [f4(v) for v in imu.rpy] + [int(imu.temperature)]
            row += ([f"{self.bms[0]:.2f}"] + [f4(v) for v in self.bms[1]]) if self.bms else [""] * 8
            row += ([f"{self.mbs[0]:.2f}"] + [f4(v) for v in self.mbs[1]]) if self.mbs else [""] * 13
            self.session.write(row)
            self.n_since_status += 1

    # ---------- 1 Hz, hilo principal ----------
    def on_timer(self):
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
                return
            s.flush()
            if time.monotonic() - self.last_status >= self.a.status_s:
                dt = time.monotonic() - self.last_status
                tm = s.meta["temp_max_arm"]
                hot = ", ".join(f"{k.replace('left_', 'L_').replace('right_', 'R_')} {v}"
                                for k, v in sorted(tm.items(), key=lambda kv: -kv[1])[:3])
                cpu = time.process_time()
                log(f"{(time.time() - s.t0) / 60:.0f} min, {self.n_since_status / dt:.0f} Hz grabados, "
                    f"SOC {self.soc}%, T máx brazos: {hot}, CPU del logger {cpu:.0f} s en total")
                self.last_status = time.monotonic()
                self.n_since_status = 0

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
    ap = argparse.ArgumentParser(description="Logger pasivo continuo del G1 (temperatura, torque, energía).")
    ap.add_argument("--out", default=os.path.expanduser("~/g1_logs"))
    ap.add_argument("--rate", type=float, default=50.0, help="Hz de grabación (LowState llega a ~500 Hz)")
    ap.add_argument("--chunk-min", type=float, default=10.0, help="minutos por archivo CSV")
    ap.add_argument("--timeout-s", type=float, default=5.0, help="segundos sin LowState para cerrar la sesión")
    ap.add_argument("--soc-jump", type=int, default=5, help="subida de SOC (%%) que se toma como batería nueva")
    ap.add_argument("--status-s", type=float, default=60.0, help="cada cuánto imprimir el estado")
    ap.add_argument("--no-gzip", dest="gzip", action="store_false", help="no comprimir los trozos cerrados")
    ap.add_argument("--net-if", default="enp62s0", help="interfaz cableada al robot (la del picker); 'auto' = autodetectar")
    ap.add_argument("--domain", type=int, default=0, help="dominio DDS (el picker usa 0)")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--bms-topic", default="rt/lf/bmsstate")
    ap.add_argument("--mbs-topic", default="rt/lf/mainboardstate")
    ap.add_argument("--nice", type=int, default=10, help="prioridad más baja para este proceso (0 = no tocar)")
    ap.add_argument("--cpu", type=int, default=None, help="fijar el proceso a este núcleo")
    args = ap.parse_args()
    if args.net_if == "auto":
        args.net_if = None
    os.makedirs(args.out, exist_ok=True)
    if args.nice:
        try:
            os.nice(args.nice)
        except OSError as e:
            log(f"no pude bajar la prioridad: {e}")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())    # kill / apagado del PC -> cierre limpio

    node = ThermalLogger(args)
    node.start()
    try:
        t_next = time.monotonic()
        t_timer = t_next
        while not stop.is_set():
            node.poll()
            t_next += node.period
            now = time.monotonic()
            if now - t_timer >= 1.0:
                t_timer = now
                node.on_timer()
            if t_next < now:                    # nos atrasamos (disco, gzip...): sin ráfagas de recuperación
                t_next = now
            stop.wait(t_next - now)
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
