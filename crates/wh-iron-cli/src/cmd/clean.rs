//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//! `iron clean` — remove build artifacts and cached snapshots.
//!
//! By default only removes generated build outputs (`*.air`, `*.metallib`,
//! any directory written by `--emit`).  Pass `--snapshots` to also wipe
//! regression baselines, or `--all` to remove everything.

use std::path::Path;

use crate::{
    CleanArgs,
    CliError,
    term::{Color, Style, paint_stderr},
};

pub fn run(args: &CleanArgs) -> Result<(), CliError> {
    let _span = tracing::info_span!("clean", snapshots = args.snapshots, all = args.all).entered();

    let mut removed = 0usize;

    // Locate project root (where iron.toml lives, or CWD as fallback).
    let root = crate::config::ConfigLoader::find_iron_toml()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));

    // ── Build artifacts ────────────────────────────────────────────────────

    // Remove iron build output directory (air intermediates).
    removed += remove_dir_if_exists(&root.join("target").join("iron-build-air"));

    // Remove any *.metallib / *.air files emitted into a user `--out` dir
    // (best-effort; we don't know the exact path the user chose, so sweep
    // the project root shallowly).
    removed += remove_files_with_ext(&root, "metallib");
    removed += remove_files_with_ext(&root, "air");

    // ── Snapshots ─────────────────────────────────────────────────────────

    if args.snapshots || args.all {
        removed += remove_dir_if_exists(&root.join(".iron-snapshots"));
    }

    // ── Summary ───────────────────────────────────────────────────────────

    if removed == 0 {
        eprintln!(
            "{}  nothing to clean",
            paint_stderr("iron clean ·", Style::new().fg(Color::Cyan).bold()),
        );
    } else {
        eprintln!(
            "{}  removed {removed} item(s)",
            paint_stderr("iron clean ·", Style::new().fg(Color::Cyan).bold()),
        );
    }

    Ok(())
}

fn remove_dir_if_exists(path: &Path) -> usize {
    if path.exists() {
        match std::fs::remove_dir_all(path) {
            Ok(()) => {
                eprintln!("  removed  {}", path.display());
                1
            },
            Err(e) => {
                eprintln!("  warning: could not remove {}: {e}", path.display());
                0
            },
        }
    } else {
        0
    }
}

/// Remove all files with the given extension directly inside `dir` (non-recursive).
fn remove_files_with_ext(dir: &Path, ext: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut n = 0;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some(ext) {
            match std::fs::remove_file(&p) {
                Ok(()) => {
                    eprintln!("  removed  {}", p.display());
                    n += 1;
                },
                Err(e) => {
                    eprintln!("  warning: could not remove {}: {e}", p.display());
                },
            }
        }
    }
    n
}
