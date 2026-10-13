"""The PMU player: sends the grid's state at 50 frames a second and times the
answers. Runs on the Jetson AGX Orin in the mini project; in the notebook it
runs in a thread on the same machine as the screener (the virtual board).

    python3 pmu_player.py --to 192.168.1.20 --port 4712 --recording Ex12.2_39_edge_recording.npz

The recording is written by the notebook (section 7): a minute of the grid
as PMUs see it. Each frame is sent on a fixed 20 ms clock whether or not the
last answer has come back - a PMU does not wait - and every answer's round
trip is measured on the player's own clock. The log is written as
``player_log.npz``.
"""
import argparse
import socket
import threading
import time

import numpy as np

import frames


def play(frames_xy, to=("127.0.0.1", 4712), period=0.02, listen_port=4713, log=None, stop=None):
    """Send every frame on a fixed clock; collect the verdicts. Returns a dict
    of arrays: seq, round trip [ms], processing [ms], insecure count."""
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", listen_port)); rx.settimeout(0.2)
    got = {}
    done = threading.Event()

    def receive():
        while not done.is_set():
            try:
                b, _ = rx.recvfrom(256)
            except socket.timeout:
                continue
            seq, t, n, ms, bits = frames.unpack_verdict(b)
            got[seq] = (1000 * (time.perf_counter() - t), ms, n)

    th = threading.Thread(target=receive, daemon=True); th.start()
    t0 = time.perf_counter()
    for seq, x in enumerate(frames_xy):
        if stop is not None and stop.is_set():
            break
        wait = t0 + seq * period - time.perf_counter()        # the 20 ms clock: sleep, then wait the last millisecond
        if wait > 0.002:
            time.sleep(wait - 0.001)
        while time.perf_counter() < t0 + seq * period:
            pass
        tx.sendto(frames.pack_frame(seq, time.perf_counter(), x), to)
    time.sleep(0.5)
    done.set(); th.join()
    tx.close(); rx.close()
    seqs = np.array(sorted(got))
    out = dict(sent=len(frames_xy), seq=seqs,
               round_trip_ms=np.array([got[s][0] for s in seqs]), processing_ms=np.array([got[s][1] for s in seqs]),
               insecure=np.array([got[s][2] for s in seqs]))
    if log:
        np.savez(log, **out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default="127.0.0.1"); ap.add_argument("--port", type=int, default=4712)
    ap.add_argument("--listen", type=int, default=4713); ap.add_argument("--recording", default="Ex12.2_39_edge_recording.npz")
    a = ap.parse_args()
    rec = np.load(a.recording)["frames"]
    r = play(rec, (a.to, a.port), listen_port=a.listen, log="player_log.npz")
    rt = r["round_trip_ms"]
    print(f"sent {r['sent']} frames, {len(rt)} answered; round trip median {np.median(rt):.2f} ms, "
          f"99th percentile {np.percentile(rt, 99):.2f} ms, over 20 ms: {(rt > 20).sum()}")
