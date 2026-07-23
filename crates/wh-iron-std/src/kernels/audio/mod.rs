//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Audio / speech front-end kernels — the audio family (see
//! `docs/specs/KERNEL_CONSOLIDATION_PLAN.md`): mel spectrogram (+ STFT window,
//! filterbank, magnitude), LSTM, the vocoder iSTFT, Snake1d activation, and
//! 1-D nearest upsampling. Migrated from the legacy `iron/`.

pub mod lstm;
pub mod mel_spectrogram;
pub mod snake1d;
pub mod upsample_nearest1d;
pub mod vocoder;
