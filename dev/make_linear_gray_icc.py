#!/usr/bin/env python3
"""Generate assets/Gray-linear-v2-prtr.icc — the default PDF/X OutputIntent profile.

A minimal ICC v2.1 MONOCHROME OUTPUT profile (device class 'prtr', colour space
GRAY, PCS XYZ) whose grayTRC is the identity curve (gamma 1.0): a file value of
0.5 means half the reflectance of bare paper. PDF/X requires the OutputIntent's
DestOutputProfile to be an OUTPUT profile, and ICC v2 is the form every PDF/X
level accepts (the print master is PDF/X-4 now; X-1a/-3, PDF 1.3, took v2
only); the profile it replaces (Gray-elle-V4-g10.icc) was a v4 'mntr' (monitor)
profile, which preflight rejects. Required tags for a v2 monochrome output
profile: desc, cprt, wtpt, kTRC (ICC.1:2001-04, 6.3.3.1).

    python3 dev/make_linear_gray_icc.py   # rewrites assets/Gray-linear-v2-prtr.icc
"""
import os
import struct


def s15f16(x):
    return struct.pack(">i", round(x * 65536))


def xyz_type(x, y, z):
    return b"XYZ " + b"\0" * 4 + s15f16(x) + s15f16(y) + s15f16(z)


def text_type(s):
    return b"text" + b"\0" * 4 + s.encode("ascii") + b"\0"


def desc_type(s):
    a = s.encode("ascii") + b"\0"
    return (b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
            + struct.pack(">II", 0, 0)            # Unicode: language code, count
            + struct.pack(">HB", 0, 0) + b"\0" * 67)  # ScriptCode: code, count, 67 bytes


def curve_identity():
    return b"curv" + b"\0" * 4 + struct.pack(">I", 0)   # count 0 = identity (gamma 1.0)


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


D50 = (0.9642, 1.0, 0.8249)
tags = [
    (b"desc", desc_type("Linear reflectance grayscale (gamma 1.0), BlobBoards")),
    (b"cprt", text_type("No copyright, use freely")),
    (b"wtpt", xyz_type(*D50)),
    (b"kTRC", curve_identity()),
]

n = len(tags)
offset = 128 + 4 + 12 * n
table, data = struct.pack(">I", n), b""
for sig, body in tags:
    table += sig + struct.pack(">II", offset + len(data), len(body))
    data += pad4(body)
size = offset + len(data)

header = (struct.pack(">I", size) + b"\0" * 4         # size, preferred CMM
          + struct.pack(">I", 0x02100000)              # version 2.1.0
          + b"prtr" + b"GRAY" + b"XYZ "                # class, colour space, PCS
          + struct.pack(">6H", 2026, 10, 2, 0, 0, 0)   # date/time (fixed: reproducible bytes)
          + b"acsp" + b"\0" * 4                        # signature, platform
          + struct.pack(">I", 0) + b"\0" * 4 + b"\0" * 4   # flags, manufacturer, model
          + b"\0" * 8                                  # attributes
          + struct.pack(">I", 1)                       # rendering intent: relative colorimetric
          + s15f16(D50[0]) + s15f16(D50[1]) + s15f16(D50[2])   # PCS illuminant
          + b"\0" * 4                                  # creator
          + b"\0" * 44)                                # reserved (v2)
assert len(header) == 128

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "Gray-linear-v2-prtr.icc")
with open(out, "wb") as f:
    f.write(header + table + data)
print(f"wrote {os.path.normpath(out)} ({size} bytes)")
