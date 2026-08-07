#!/usr/bin/env python3
"""CI lint: every kernel param must be bound in standalone-test dispatches.

Name-bound dispatch resolves a missing buffer name to a pooled,
non-zero-initialized allocation — silently. A harness that predates a
kernel's newest params reads garbage through the unbound slot; whether
it passes depends on allocator luck (this corrupted CI runs from the
gdn family once: state_planes/planes_enabled, fixed in the commit that
introduced this lint). This check makes the class structural.

Validated against that known positive (the gdn state_planes/planes_enabled
incident) before being wired into CI.
"""
import re, glob, os, sys, subprocess

def extract_kernels(text):
    out = {}
    for m in re.finditer(r'fn\s+(iron_[a-z0-9_]+)\s*(?:<[^>]*>)?\s*\(', text):
        name = m.group(1)
        i = m.end(); depth = 1; buf = []
        while i < len(text) and depth:
            c = text[i]
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if depth: buf.append(c)
            i += 1
        blob = re.sub(r'//[^\n]*', '', ''.join(buf))
        pnames = set(re.findall(r'(?:mut\s+)?([a-z0-9_]+)\s*:\s*(?:Tensor|&)', blob))
        if pnames: out[name] = pnames
    return out

def main():
    root = 'crates/wh-iron-std'
    src = {}
    for f in glob.glob(f'{root}/src/kernels/**/*.rs', recursive=True):
        src.update(extract_kernels(open(f).read()))
    flagged = []
    for tf in sorted(glob.glob(f'{root}/tests/*.rs')):
        t = open(tf).read()
        keys = set(re.findall(r'\.insert\(\s*"([a-z0-9_]+)"', t))
        if not keys: continue
        for k in set(re.findall(r'(iron_[a-z0-9_]+)::kernel_ir_for', t)):
            if k in src:
                miss = src[k] - keys
                if miss: flagged.append((os.path.basename(tf), k, sorted(miss)))
    for tf, k, miss in flagged:
        print(f"UNBOUND: {tf} dispatches {k} without binding {miss}", file=sys.stderr)
    print(f"checked {len(src)} kernels; {len(flagged)} unbound-binding site(s)")
    return 1 if flagged else 0

if __name__ == '__main__':
    sys.exit(main())
