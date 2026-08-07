#!/usr/bin/env python3
"""Report-only lint: flag `iron_*` kernels with zero test or external coverage.

The RST oracle-coverage sweep (2026-08) found kernels that ship in
`crates/wh-iron-std/src/kernels/**` with no `#[test_kernel]`, no standalone
`tests/*.rs` dispatch, and no reference from the compiled Swift manifest
that the production caller (the sibling Swift package) uses to look up
kernels by name. `iron_conv_roll` and `iron_kv_append` were the two
confirmed cases at the time this script was written — genuinely dead code
or genuinely untested production code, either way worth a standing flag
so a kernel doesn't silently rot in that state indefinitely.

This is intentionally a REPORT, not a gate: exit code is always 0. The
repo's convention here is keep-but-flag, not delete-on-sight or fail-CI —
a flagged kernel might be legitimately WIP, might be exercised only via a
downstream integration test this script can't see, or might just not have
been ported to `#[test_kernel]` yet. A human reads the report and decides.

## Detection

1. Extract every `pub fn iron_<name>` kernel definition under
   `crates/wh-iron-std/src/kernels/**/*.rs` (same source-scanning approach
   as `check_test_buffer_bindings.py` — plain regex, no AST parser).
2. For each kernel, decide if it's TEMPLATED — produced by
   `#[kernel(variants(...))]` rather than a plain `#[kernel]` — by scanning
   the text between the `#[kernel` attribute and the `pub fn` line for the
   literal `variants(`. A templated kernel's Rust-source name is not the
   name any caller actually writes: `#[kernel(variants(BITS = [4, 8],
   suffix = "int{BITS}"))] pub fn iron_quantize_kv<T>(...)` compiles to
   `iron_quantize_kv_int4` / `iron_quantize_kv_int8`, and callers reference
   THOSE names, never the literal `iron_quantize_kv` source identifier.
   Exact whole-word matching would flag every templated kernel as orphaned
   by construction, so templated kernels use PREFIX matching instead (see
   below) — this is a best-effort compromise, not perfect precision; the
   value here is the report, not exact reconstruction of every generated
   variant name.
3. TEST-REFERENCE check: does the kernel's name (word-boundary for plain
   kernels, prefix for templated ones — see below) appear more than once
   across the concatenation of every file under
   `crates/wh-iron-std/src/kernels/**/*.rs` PLUS every
   `crates/wh-iron-std/tests/*.rs` file? ("More than once" because the
   kernel's own `pub fn` definition line is always one occurrence — a
   second occurrence means SOMETHING references it, whether that's its own
   `kernel_tests` module's `use super::{...}` import list or a standalone
   integration test.)
4. EXTERNAL-REFERENCE check (optional): if a compiled Swift manifest JSON
   is supplied (`{"kernels": [{"name": "iron_x_f32", ...}, ...]}`), does
   any manifest kernel name start with the Iron kernel's name (prefix
   match handles the per-dtype suffix, e.g. `iron_conv_roll` -> matches
   `iron_conv_roll_f32`)? Skipped entirely (treated as "no external
   reference found") when no manifest path is given, so this script has no
   hard dependency on the sibling Swift repo and still runs standalone.
5. A kernel is ORPHANED if it has ZERO test references AND (no manifest
   was given OR zero external-reference matches).

## Templated-kernel name matching

For a templated kernel, compute `prefix = name.split('BITS')[0]` (this is
just `name` unchanged when the literal string `BITS` never appears in it,
which is the common case — e.g. `iron_quantize_kv` with
`suffix = "int{BITS}"` appended entirely outside the base name). Both the
test-reference and external-reference checks then use `prefix` with
substring / `startswith` matching instead of exact word-boundary matching.
This handles the two templated-name shapes this codebase actually uses:
literal `BITS` embedded in the name (`iron_aura_encode_intBITS`) and BITS
appended only via the `suffix` macro argument, with none of it echoed into
the literal Rust source identifier (`iron_quantize_kv`).

Usage:
    python3 scripts/check_orphan_kernels.py [manifest.json]
"""
import glob
import re
import sys

KERNEL_DEF_RE = re.compile(
    r'#\[kernel(?P<attr>.*?)pub\s+fn\s+(?P<name>iron_[a-z0-9_]+)',
    re.DOTALL,
)


def read_all(paths):
    """Read every path in `paths`, returning [(path, text), ...]."""
    out = []
    for p in sorted(paths):
        with open(p, encoding='utf-8') as f:
            out.append((p, f.read()))
    return out


def extract_kernel_defs(kernel_files):
    """Return [(name, file, line, is_templated), ...] for every `pub fn
    iron_*` kernel definition, in the order first seen."""
    defs = []
    for path, text in kernel_files:
        for m in KERNEL_DEF_RE.finditer(text):
            name = m.group('name')
            attr = m.group('attr')
            is_templated = 'variants(' in attr
            line = text.count('\n', 0, m.start('name')) + 1
            defs.append((name, path, line, is_templated))
    return defs


def match_key(name, is_templated):
    """The string used to search for references to `name` — the full name
    for a plain kernel, or the pre-`BITS` prefix for a templated one (see
    the module docstring's "Templated-kernel name matching" section)."""
    if is_templated:
        return name.split('BITS')[0]
    return name


def count_test_references(key, is_templated, corpus_text):
    """Occurrences of `key` in `corpus_text` — word-boundary for a plain
    kernel name, plain substring for a templated-kernel prefix (a
    templated kernel's callers never write the literal source identifier,
    only a dtype/variant-suffixed derivative of it)."""
    if is_templated:
        return corpus_text.count(key)
    return len(re.findall(r'\b' + re.escape(key) + r'\b', corpus_text))


def has_external_reference(key, manifest_names):
    if manifest_names is None:
        return False
    return any(mname.startswith(key) for mname in manifest_names)


def load_manifest_names(manifest_path):
    if manifest_path is None:
        return None
    import json
    with open(manifest_path, encoding='utf-8') as f:
        data = json.load(f)
    return {k['name'] for k in data.get('kernels', [])}


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else None
    manifest_names = load_manifest_names(manifest_path)

    kernel_files = read_all(glob.glob('crates/wh-iron-std/src/kernels/**/*.rs', recursive=True))
    test_files = read_all(glob.glob('crates/wh-iron-std/tests/*.rs'))
    corpus_text = ''.join(t for _, t in kernel_files) + ''.join(t for _, t in test_files)

    defs = extract_kernel_defs(kernel_files)

    orphans = []
    for name, path, line, is_templated in defs:
        key = match_key(name, is_templated)
        n_test_refs = count_test_references(key, is_templated, corpus_text)
        # The kernel's own definition line is always one occurrence.
        has_test_ref = n_test_refs > 1
        has_ext_ref = has_external_reference(key, manifest_names)
        if not has_test_ref and not has_ext_ref:
            orphans.append((name, path, line))

    print(f'checked {len(defs)} kernel definitions', file=sys.stderr)
    if manifest_names is not None:
        print(f'manifest: {len(manifest_names)} compiled kernel names loaded', file=sys.stderr)
    else:
        print('manifest: none given — external-reference check skipped', file=sys.stderr)

    if orphans:
        print(f'{len(orphans)} orphaned kernel(s) (no test reference, no external reference):')
        for name, path, line in sorted(orphans):
            print(f'  {name}  ({path}:{line})')
    else:
        print('no orphaned kernels found')

    # Report-only: this lint never fails CI (see module docstring).
    return 0


if __name__ == '__main__':
    sys.exit(main())
