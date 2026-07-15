#!/usr/bin/env python3
"""Embed a binary file as a C array with C linkage.

Usage: bin2c.py <input-file> <symbol-name>  > out.c

Emits:
    extern "C" const unsigned char <symbol>[] = { 0x.., ... };
    extern "C" const unsigned long <symbol>_len = <N>;

Matches cuda_iface.h's `extern "C" const unsigned char ps_hip_co[];`, which
is handed straight to hipModuleLoadData (the loader reads the code object's
size from its ELF header, so the _len symbol is emitted for completeness but
is not required by the current call site).
"""
import sys


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: bin2c.py <input-file> <symbol-name>\n")
        return 2
    path, name = sys.argv[1], sys.argv[2]
    with open(path, "rb") as f:
        data = f.read()

    out = sys.stdout
    out.write(f'extern "C" const unsigned char {name}[] = {{\n')
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        out.write("  " + ",".join(f"0x{b:02x}" for b in chunk) + ",\n")
    out.write("};\n")
    out.write(f'extern "C" const unsigned long {name}_len = {len(data)}ul;\n')
    return 0


if __name__ == "__main__":
    sys.exit(main())
