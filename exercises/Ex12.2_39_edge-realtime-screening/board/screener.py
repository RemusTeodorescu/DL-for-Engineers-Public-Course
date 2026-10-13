"""The screener: takes the newest PMU frame, screens all 35 contingencies
with the surrogate, and answers. Runs on the Jetson AGX Thor in the mini
project; in the notebook it runs in a thread beside the player.

    python3 screener.py --reply-to 192.168.1.10 --weights Ex12.2_39_edge_surrogate.pt --device cuda

A frame that arrives while the last one is being screened is not queued:
the screener always takes the newest and drops the older ones, because an
answer about a stale state is worth less than no answer. Dropped frames are
counted. The log is written as ``screener_log.npz``.
"""
import argparse
import socket
import time

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # problem.py and grid39.py, one folder up

import frames
import problem as pb
import surrogate


def serve(net, g, listen_port=4712, reply_to=("127.0.0.1", 4713), stop=None, log=None, idle_timeout=2.0, first_timeout=60.0):
    screen = surrogate.Screen(net, g)
    sync = (lambda: torch.cuda.synchronize()) if g.device.type == "cuda" else (lambda: None)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", listen_port)); rx.settimeout(first_timeout)   # wait up to a minute for the first frame
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    processing, dropped, seqs = [], 0, []
    screen(np.stack([g.np["d0"], 0 * g.np["d0"], g.np["E0"]], -1)); sync()   # warm-up on the equilibrium
    while stop is None or not stop.is_set():
        try:
            b, _ = rx.recvfrom(1024)
        except socket.timeout:
            break                                              # the player has stopped
        rx.setblocking(False)
        while True:                                            # keep only the newest frame
            try:
                b, _ = rx.recvfrom(1024); dropped += 1
            except BlockingIOError:
                break
        rx.setblocking(True); rx.settimeout(idle_timeout)
        seq, t, x = frames.unpack_frame(b)
        t0 = time.perf_counter()
        worst = screen(x); sync()
        insecure = (worst > np.pi).cpu().numpy()
        ms = 1000 * (time.perf_counter() - t0)
        tx.sendto(frames.pack_verdict(seq, t, insecure, ms), reply_to)
        processing.append(ms); seqs.append(seq)
    rx.close(); tx.close()
    out = dict(seq=np.array(seqs), processing_ms=np.array(processing), dropped=dropped)
    if log:
        np.savez(log, **out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4712); ap.add_argument("--reply-to", default="127.0.0.1")
    ap.add_argument("--reply-port", type=int, default=4713); ap.add_argument("--weights", default="Ex12.2_39_edge_surrogate.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true", help="run the surrogate in 16-bit floats")
    a = ap.parse_args()
    dtype = torch.float16 if a.half else torch.float32
    net = surrogate.load(a.weights, a.device, dtype)
    g = pb.Grid(1.0, (), (31, 33)).to(a.device, torch.float32)          # the state the recording was made in
    r = serve(net, g, a.port, (a.reply_to, a.reply_port), log="screener_log.npz")
    p = r["processing_ms"]
    print(f"screened {len(p)} frames, dropped {r['dropped']}; processing median {np.median(p):.2f} ms, "
          f"99th percentile {np.percentile(p, 99):.2f} ms")
