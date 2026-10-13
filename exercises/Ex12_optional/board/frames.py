"""The two packets of the real-time loop, sent by UDP.

A PMU frame (player -> screener): sequence number, the player's send time,
and the state of the 39 buses - angle [rad], frequency [rad/s], E [p.u.] -
as 32-bit floats. (A real PMU stream is IEEE C37.118; this is the same idea,
stripped to what the screen reads.)

A verdict (screener -> player): the frame's sequence number and send time,
echoed back so the player measures the round trip on its own clock, the
number of insecure contingencies, the screener's own processing time, and
one bit per contingency.
"""
import struct
import numpy as np

N_BUS, N_CASE = 39, 35
FRAME = struct.Struct("<Id")                 # seq, send time
VERDICT = struct.Struct("<IdHf")             # seq, echoed send time, insecure count, processing time [ms]


def pack_frame(seq, t, x):
    return FRAME.pack(seq, t) + np.asarray(x, dtype="<f4").reshape(N_BUS * 3).tobytes()


def unpack_frame(b):
    seq, t = FRAME.unpack_from(b)
    return seq, t, np.frombuffer(b, dtype="<f4", offset=FRAME.size).reshape(N_BUS, 3)


def pack_verdict(seq, t, insecure, ms):
    bits = np.packbits(np.asarray(insecure, dtype=bool)).tobytes()
    return VERDICT.pack(seq, t, int(np.sum(insecure)), ms) + bits


def unpack_verdict(b):
    seq, t, n, ms = VERDICT.unpack_from(b)
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8, offset=VERDICT.size))[:N_CASE].astype(bool)
    return seq, t, n, ms, bits
