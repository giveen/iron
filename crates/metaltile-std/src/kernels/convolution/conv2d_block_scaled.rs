//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled **quantized-weight 2D convolution** — the weight-quantized
//! counterpart of `ffai/conv2d.rs`.
//!
//! A conv2d output element is a dot product over the `in_ch × kh × kw`
//! receptive field against a filter row, so the filter `[out_ch, in_ch, kh,
//! kw]` is a genuine quantizable parameter. We treat it as a 2-D matrix
//! `[out_ch, C]` with `C = in_ch · kh · kw` — the per-output-channel
//! contraction — block-scaled along `C` in the spec formats (mxfp4 / nvfp4 /
//! mxfp8 e4m3+e5m2 / nvfp8 + legacy fp4/fp8 + symmetric int8 + sub-byte
//! int2..int6 / mxint2..mxint6 + mxint8).
//!
//! For an output channel `oc` and a tap `(ic, ky, kx)` the contraction index
//! is `col = (ic·kh + ky)·kw + kx = ic·kh·kw + ky·kw + kx`. The dense filter
//! value `weight[((oc·in_ch+ic)·kh+ky)·kw+kx]` becomes
//! `element_decode(code[oc, col]) · block_scale[oc, col/block_size]` (× global
//! for nvfp4). 4-bit codes are packed `[out_ch, C/8]` u32 (8 nibbles/word, code
//! at word `oc·(C/8)+col/8`, shift `(col%8)·4`); 8-bit codes are `[out_ch, C]`
//! u8 (byte at `oc·C+col`). Sub-byte integers (int2..6 / mxint2..6) are a tight
//! LSB-first bit-stream, per output-channel row word-aligned: row `oc` starts at
//! word `oc·(C·bits/32)`, code `col` at bit `col·bits` (straddle-aware read).
//! Only the filter is quantized — the per-channel `bias` stays `T`.
//!
//! Geometry is **identical** to the dense `conv2d_generic`: **Grid3D**, one
//! thread per output element (`program_id::<0>()` = flat
//! `((n·out_ch+oc)·out_h+oh)·out_w+ow`), the same stride / padding / dilation
//! receptive-field walk in the padded input frame, fp32 accumulation, padding
//! taps clamped to contribute zero. `C` is a multiple of `block_size` (4-bit
//! `block_size` a multiple of 8). fp8_e4m3 reuses the nvfp8 kernel. Codegen-only;
//! correctness pinned by the in-source `#[test_kernel]`s vs a
//! `quant::format::dequant` oracle running the dense conv2d math.

use metaltile::kernel;
/// Quantized-weight conv2d, folded over the 28-format axis (§7). Geometry is the
/// dense `conv2d_generic` receptive-field walk; only the per-column filter
/// decode + per-block scale fold onto the `(BITS, WDEC, SKIND)` co-vars (buffer
/// types by `(WT, ST)` — see `gemm/block_scaled_matmul` for the legend). `col`
/// is the contraction index `ic·kh·kw + ky·kw + kx`. Decodes through
/// `kernels/primitives.rs`. Produces `mt_<FMT>_conv2d`.
#[kernel(variants(
    (FMT,          BITS,  WT,  ST,  WDEC, SKIND) = [
        (mxfp4,        4u32, u32, u8,  0u32, 0u32),
        (nvfp4,        4u32, u32, u8,  0u32, 1u32),
        (fp4,          4u32, u32, f32, 0u32, 2u32),
        (fp4_f16,      4u32, u32, f16, 0u32, 2u32),
        (int2,         2u32, u32, f32, 1u32, 2u32),
        (int3,         3u32, u32, f32, 1u32, 2u32),
        (int4,         4u32, u32, f32, 1u32, 2u32),
        (int5,         5u32, u32, f32, 1u32, 2u32),
        (int6,         6u32, u32, f32, 1u32, 2u32),
        (mxint2,       2u32, u32, u8,  1u32, 0u32),
        (mxint3,       3u32, u32, u8,  1u32, 0u32),
        (mxint4,       4u32, u32, u8,  1u32, 0u32),
        (mxint5,       5u32, u32, u8,  1u32, 0u32),
        (mxint6,       6u32, u32, u8,  1u32, 0u32),
        (int2_f16,     2u32, u32, f16, 1u32, 2u32),
        (int3_f16,     3u32, u32, f16, 1u32, 2u32),
        (int4_f16,     4u32, u32, f16, 1u32, 2u32),
        (int5_f16,     5u32, u32, f16, 1u32, 2u32),
        (int6_f16,     6u32, u32, f16, 1u32, 2u32),
        (mxfp8_e4m3,   8u32, u8,  u8,  2u32, 0u32),
        (mxfp8_e5m2,   8u32, u8,  u8,  3u32, 0u32),
        (mxint8,       8u32, u8,  u8,  4u32, 0u32),
        (nvfp8,        8u32, u8,  f32, 2u32, 2u32),
        (fp8_e5m2,     8u32, u8,  f32, 3u32, 2u32),
        (int8,         8u32, u8,  f32, 4u32, 2u32),
        (nvfp8_f16,    8u32, u8,  f16, 2u32, 2u32),
        (fp8_e5m2_f16, 8u32, u8,  f16, 3u32, 2u32),
        (int8_f16,     8u32, u8,  f16, 4u32, 2u32),
    ],
    suffix = "{FMT}_conv2d",
))]
#[allow(clippy::too_many_arguments)]
pub fn mt<T>(
    input: Tensor<T>,
    weight: Tensor<WT>,
    scales: Tensor<ST>,
    bias: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] in_ch: u32,
    #[constexpr] in_h: u32,
    #[constexpr] in_w: u32,
    #[constexpr] out_ch: u32,
    #[constexpr] out_h: u32,
    #[constexpr] out_w: u32,
    #[constexpr] kh: u32,
    #[constexpr] kw: u32,
    #[constexpr] stride_h: u32,
    #[constexpr] stride_w: u32,
    #[constexpr] pad_h: u32,
    #[constexpr] pad_w: u32,
    #[constexpr] block_size: u32,
    #[constexpr(only_when = "SKIND == 1u32")] global: f32,
) {
    let idx = program_id::<0>();
    let ow = idx % out_w;
    let t1 = idx / out_w;
    let oh = t1 % out_h;
    let t2 = t1 / out_h;
    let oc = t2 % out_ch;
    let n = t2 / out_ch;
    let ph0 = oh * stride_h;
    let pw0 = ow * stride_w;
    let input_plane = in_h * in_w;
    let in_n_stride = in_ch * input_plane;
    let contraction = in_ch * kh * kw;
    let n_blocks = contraction / block_size;
    let w_row_pack = oc * (contraction / 8u32);
    let w_row_word = oc * (contraction * BITS / 32u32);
    let w_row_byte = oc * contraction;
    let w_row_blk = oc * n_blocks;
    let half = 1u32 << (BITS - 1u32);
    let full = (1u32 << BITS).cast::<f32>();
    let mut acc = load(bias[oc]).cast::<f32>();
    for ic in range(0u32, in_ch, 1u32) {
        let in_ic_base = n * in_n_stride + ic * input_plane;
        let col_ic = ic * kh * kw;
        for ky in range(0u32, kh, 1u32) {
            let ph = ph0 + ky;
            let row_ok = (ph >= pad_h) & (ph < pad_h + in_h);
            let ih = select(row_ok, ph - pad_h, 0u32);
            for kx in range(0u32, kw, 1u32) {
                let pw = pw0 + kx;
                let col_ok = (pw >= pad_w) & (pw < pad_w + in_w);
                let valid = row_ok & col_ok;
                let iw = select(col_ok, pw - pad_w, 0u32);
                let pix = load(input[in_ic_base + ih * in_w + iw]).cast::<f32>();
                let pix_m = select(valid, pix, 0.0f32);
                let col = col_ic + ky * kw + kx;
                let elem = if WDEC == 0u32 {
                    mt_decode_e2m1((load(weight[w_row_pack + col / 8u32]) >> ((col % 8u32) * 4u32))
                        & 0xFu32)
                } else if WDEC == 1u32 {
                    let bit_off = col * BITS;
                    let word_idx = bit_off / 32u32;
                    let bit_in_w = bit_off & 31u32;
                    let bits_in_w0 = 32u32 - bit_in_w;
                    let lo_bits = select(bits_in_w0 >= BITS, BITS, bits_in_w0);
                    let spill = BITS - lo_bits;
                    let w0 = load(weight[w_row_word + word_idx]);
                    let w1 = load(
                        weight[w_row_word + select(spill > 0u32, word_idx + 1u32, word_idx)],
                    );
                    let q = mt_unpack_nbit(w0, w1, bit_in_w, lo_bits, spill);
                    let qf = q.cast::<f32>();
                    select(q >= half, qf - full, qf)
                } else {
                    let raw = load(weight[w_row_byte + col]).cast::<u32>();
                    if WDEC == 2u32 {
                        mt_decode_e4m3(raw)
                    } else if WDEC == 3u32 {
                        mt_decode_e5m2(raw)
                    } else {
                        mt_decode_int8(raw)
                    }
                };
                let sraw = load(scales[w_row_blk + col / block_size]);
                let scale = if SKIND == 0u32 {
                    exp2(sraw.cast::<f32>() - 127.0f32)
                } else if SKIND == 1u32 {
                    mt_decode_e4m3(sraw.cast::<u32>()) * global
                } else {
                    sraw.cast::<f32>()
                };
                acc = acc + pix_m * (elem * scale);
            }
        }
    }
    store(out[idx], acc.cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::{
        quant::format::QFormat,
        utils::{pack_f32, unpack_f32},
    };

    /// Deterministic ramp identical to the dense conv2d helper: a bounded
    /// zig-zag so f16/bf16 stay in range.
    fn ramp(n: usize, period: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp).collect()
    }

    /// Direct 2D conv oracle (NCHW input, OIHW weight), groups=1, dilation=1.
    /// Padding taps contribute zero — the SAME dense math as conv2d.rs's
    /// `naive_conv2d`, run over the *dequantized* filter. All f32.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn naive_conv2d(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        // Quantized filter is laid out as the 2-D matrix [out_ch, C] with
        // C = in_ch*kh*kw and col = (ic*kh + ky)*kw + kx, so the dequantized
        // weight row `oc` is contiguous over `col`.
        let contraction = in_ch * kh * kw;
        let mut out = vec![0.0f32; batch * out_ch * out_h * out_w];
        for n in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias[oc];
                        for ic in 0..in_ch {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ph = oh * stride_h + ky;
                                    let pw = ow * stride_w + kx;
                                    if ph < pad_h
                                        || ph >= pad_h + in_h
                                        || pw < pad_w
                                        || pw >= pad_w + in_w
                                    {
                                        continue;
                                    }
                                    let ih = ph - pad_h;
                                    let iw = pw - pad_w;
                                    let in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                                    let col = (ic * kh + ky) * kw + kx;
                                    let w_idx = oc * contraction + col;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        let o_idx = ((n * out_ch + oc) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        out
    }

    /// QFormat-parametrized setup: quantize the [out_ch, C] filter via the
    /// shared codec, dequantize for the oracle, and run the dense conv2d math.
    /// Mirrors conv2d.rs's `conv2d_setup` grid + KernelMode exactly.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_setup(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dt: DType,
    ) -> TestSetup {
        let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        // Contraction C = in_ch*kh*kw — the quantized filter is [out_ch, C].
        let contraction = in_ch * kh * kw;
        let input_f = ramp(batch * in_ch * in_h * in_w, 13, 6.0);
        let bias_f = ramp(out_ch, 5, 2.0);
        // Quantize the [out_ch, C] filter via the shared codec.
        let w_f = ramp(out_ch * contraction, 11, 4.0);
        let p = crate::quant::format::pack(fmt, &w_f, out_ch, contraction);
        let wdq = crate::quant::format::dequant(fmt, &p, out_ch, contraction);
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let bias = unpack_f32(&pack_f32(&bias_f, dt), dt);
        // Oracle: dense conv2d over the dequantized filter row [out_ch, C].
        let expected = naive_conv2d(
            &input, &wdq, &bias, batch, in_ch, in_h, in_w, out_ch, kh, kw, stride_h, stride_w,
            pad_h, pad_w,
        );
        // 8-bit codes bind as one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) binds as packed u32 words. FP32
        // scales bind as f32; E8M0/E4M3 scales as one byte. Both axes are driven
        // off the format so new integer formats pick up the right buffer types.
        let weight_dt = if fmt.element_bits() == 8 { DType::U8 } else { DType::U32 };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let mut s = TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("weight", p.codes, weight_dt))
            .input(TestBuffer::from_vec("scales", p.scales, scales_dt))
            .input(TestBuffer::from_vec("bias", pack_f32(&bias_f, dt), dt))
            .input(TestBuffer::zeros("out", n_out, dt))
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", pad_h as u32)
            .constexpr("pad_w", pad_w as u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", p.global);
        }
        s.expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt)).grid_1d(n_out, 256)
    }

    // One correctness test per QFormat via the shared `conv2d_setup` helper —
    // mirrors the `conv2d_bench_fmt!` benches instead of 30 hand-written fns.
    // Shape: in_ch=4, 4×4 kernel → C=64 (÷ 16/32/64); 8×8 image, stride/dilation
    // 1, pad 1, out_ch=8 — exercises the in-kernel padding clamp.
    macro_rules! conv2d_test_fmt {
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = [5e-3, 5e-2, 2e-1])]
            fn $fn(dt: DType) -> TestSetup {
                conv2d_setup($kernel(dt), $fmt, 1, 4, 8, 8, 8, 4, 4, 1, 1, 1, 1, dt)
            }
        };
    }
    conv2d_test_fmt!(test_mxfp4_conv2d, mt_mxfp4_conv2d::kernel_ir_for, QFormat::Mxfp4);
    conv2d_test_fmt!(test_nvfp4_conv2d, mt_nvfp4_conv2d::kernel_ir_for, QFormat::Nvfp4);
    conv2d_test_fmt!(test_fp4_conv2d, mt_fp4_conv2d::kernel_ir_for, QFormat::Fp4);
    conv2d_test_fmt!(test_mxfp8_e4m3_conv2d, mt_mxfp8_e4m3_conv2d::kernel_ir_for, QFormat::Mxfp8E4);
    conv2d_test_fmt!(test_mxfp8_e5m2_conv2d, mt_mxfp8_e5m2_conv2d::kernel_ir_for, QFormat::Mxfp8E5);
    conv2d_test_fmt!(test_fp8_e5m2_conv2d, mt_fp8_e5m2_conv2d::kernel_ir_for, QFormat::Fp8E5m2);
    conv2d_test_fmt!(test_nvfp8_conv2d, mt_nvfp8_conv2d::kernel_ir_for, QFormat::Nvfp8);
    conv2d_test_fmt!(test_fp8_e4m3_conv2d, mt_nvfp8_conv2d::kernel_ir_for, QFormat::Fp8E4m3);
    conv2d_test_fmt!(test_int8_conv2d, mt_int8_conv2d::kernel_ir_for, QFormat::Int8);
    conv2d_test_fmt!(test_int2_conv2d, mt_int2_conv2d::kernel_ir_for, QFormat::Int2);
    conv2d_test_fmt!(test_int3_conv2d, mt_int3_conv2d::kernel_ir_for, QFormat::Int3);
    conv2d_test_fmt!(test_int4_conv2d, mt_int4_conv2d::kernel_ir_for, QFormat::Int4);
    conv2d_test_fmt!(test_int5_conv2d, mt_int5_conv2d::kernel_ir_for, QFormat::Int5);
    conv2d_test_fmt!(test_int6_conv2d, mt_int6_conv2d::kernel_ir_for, QFormat::Int6);
    conv2d_test_fmt!(test_mxint2_conv2d, mt_mxint2_conv2d::kernel_ir_for, QFormat::Mxint2);
    conv2d_test_fmt!(test_mxint3_conv2d, mt_mxint3_conv2d::kernel_ir_for, QFormat::Mxint3);
    conv2d_test_fmt!(test_mxint4_conv2d, mt_mxint4_conv2d::kernel_ir_for, QFormat::Mxint4);
    conv2d_test_fmt!(test_mxint5_conv2d, mt_mxint5_conv2d::kernel_ir_for, QFormat::Mxint5);
    conv2d_test_fmt!(test_mxint6_conv2d, mt_mxint6_conv2d::kernel_ir_for, QFormat::Mxint6);
    conv2d_test_fmt!(test_mxint8_conv2d, mt_mxint8_conv2d::kernel_ir_for, QFormat::Mxint8);
    conv2d_test_fmt!(test_nvfp8_f16_conv2d, mt_nvfp8_f16_conv2d::kernel_ir_for, QFormat::Nvfp8F16);
    conv2d_test_fmt!(
        test_fp8_e4m3_f16_conv2d,
        mt_nvfp8_f16_conv2d::kernel_ir_for,
        QFormat::Fp8E4m3F16
    );
    conv2d_test_fmt!(test_fp4_f16_conv2d, mt_fp4_f16_conv2d::kernel_ir_for, QFormat::Fp4F16);
    conv2d_test_fmt!(
        test_fp8_e5m2_f16_conv2d,
        mt_fp8_e5m2_f16_conv2d::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    conv2d_test_fmt!(test_int2_f16_conv2d, mt_int2_f16_conv2d::kernel_ir_for, QFormat::Int2F16);
    conv2d_test_fmt!(test_int3_f16_conv2d, mt_int3_f16_conv2d::kernel_ir_for, QFormat::Int3F16);
    conv2d_test_fmt!(test_int4_f16_conv2d, mt_int4_f16_conv2d::kernel_ir_for, QFormat::Int4F16);
    conv2d_test_fmt!(test_int5_f16_conv2d, mt_int5_f16_conv2d::kernel_ir_for, QFormat::Int5F16);
    conv2d_test_fmt!(test_int6_f16_conv2d, mt_int6_f16_conv2d::kernel_ir_for, QFormat::Int6F16);
    conv2d_test_fmt!(test_int8_f16_conv2d, mt_int8_f16_conv2d::kernel_ir_for, QFormat::Int8F16);
}

/// Decode-shape benches: a realistic conv (in_ch=64, out_ch=128, 4×4 kernel →
/// C = 1024, divisible by all block sizes). Grid3D, one thread per output
/// element; bytes_moved counts weight + scales + input + output streams.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;
    use crate::quant::format::QFormat;

    #[allow(clippy::too_many_arguments)]
    fn conv2d_bench(
        kernel: Kernel,
        fmt: QFormat,
        batch: usize,
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        out_ch: usize,
        kh: usize,
        kw: usize,
        stride_h: usize,
        stride_w: usize,
        dt: DType,
    ) -> BenchSetup {
        let out_h = (in_h - kh) / stride_h + 1;
        let out_w = (in_w - kw) / stride_w + 1;
        let n_out = batch * out_ch * out_h * out_w;
        let contraction = in_ch * kh * kw;
        // 8-bit codes are one uchar each; every sub-byte width (4-bit nibble
        // packs + int2/3/5/6 tight bit-streams) tight-bit-packs into u32 words.
        let n_codes = out_ch * contraction;
        let (codes_len, codes_dt) = if fmt.element_bits() == 8 {
            (n_codes, DType::U8)
        } else {
            (crate::quant::format::bitstream_words(n_codes, fmt.element_bits()), DType::U32)
        };
        let scales_dt = match fmt.scale_kind() {
            crate::quant::format::ScaleKind::F32 => DType::F32,
            crate::quant::format::ScaleKind::F16 => DType::F16,
            _ => DType::U8,
        };
        let n_blocks = out_ch * (contraction / fmt.block_size());
        let sz = dt.size_bytes();
        let bytes = codes_len * codes_dt.size_bytes()
            + n_blocks * scales_dt.size_bytes()
            + batch * in_ch * in_h * in_w * sz
            + out_ch * sz
            + n_out * sz;
        let mut s = BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", batch * in_ch * in_h * in_w, dt))
            .buffer(BenchBuffer::random("weight", codes_len, codes_dt))
            .buffer(BenchBuffer::random("scales", n_blocks, scales_dt))
            .buffer(BenchBuffer::random("bias", out_ch, dt))
            .buffer(BenchBuffer::zeros("out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("in_ch", in_ch as u32)
            .constexpr("in_h", in_h as u32)
            .constexpr("in_w", in_w as u32)
            .constexpr("out_ch", out_ch as u32)
            .constexpr("out_h", out_h as u32)
            .constexpr("out_w", out_w as u32)
            .constexpr("kh", kh as u32)
            .constexpr("kw", kw as u32)
            .constexpr("stride_h", stride_h as u32)
            .constexpr("stride_w", stride_w as u32)
            .constexpr("pad_h", 0u32)
            .constexpr("pad_w", 0u32)
            .constexpr("block_size", fmt.block_size() as u32);
        if matches!(fmt, QFormat::Nvfp4) {
            s = s.constexpr("global", 1.0f32);
        }
        s.grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
            // 2 * n_out * C; C = in_ch*kh*kw is the per-output contraction.
            .flops(2 * n_out as u64 * contraction as u64)
            .with_shape_label(format!(
                "{} co={out_ch} ho={out_h} wo={out_w} C={contraction}",
                fmt.name()
            ))
    }

    macro_rules! conv2d_bench_fmt {
        ($fn:ident, $kernel:path, $fmt:expr) => {
            #[bench(dtypes = [f32, f16, bf16])]
            fn $fn(dt: DType) -> BenchSetup {
                // in_ch=64, out_ch=128, 4×4 kernel → C=1024 (÷ 16/32/64).
                conv2d_bench($kernel(dt), $fmt, 1, 64, 56, 56, 128, 4, 4, 1, 1, dt)
            }
        };
    }
    conv2d_bench_fmt!(bench_mxfp4, mt_mxfp4_conv2d::kernel_ir_for, QFormat::Mxfp4);
    conv2d_bench_fmt!(bench_nvfp4, mt_nvfp4_conv2d::kernel_ir_for, QFormat::Nvfp4);
    conv2d_bench_fmt!(bench_fp4, mt_fp4_conv2d::kernel_ir_for, QFormat::Fp4);
    conv2d_bench_fmt!(bench_mxfp8_e4m3, mt_mxfp8_e4m3_conv2d::kernel_ir_for, QFormat::Mxfp8E4);
    conv2d_bench_fmt!(bench_mxfp8_e5m2, mt_mxfp8_e5m2_conv2d::kernel_ir_for, QFormat::Mxfp8E5);
    conv2d_bench_fmt!(bench_fp8_e5m2, mt_fp8_e5m2_conv2d::kernel_ir_for, QFormat::Fp8E5m2);
    conv2d_bench_fmt!(bench_nvfp8, mt_nvfp8_conv2d::kernel_ir_for, QFormat::Nvfp8);
    conv2d_bench_fmt!(bench_int8, mt_int8_conv2d::kernel_ir_for, QFormat::Int8);
    // Symmetric sub-byte ints (FP32 group scale) + MXINT (E8M0 block scale) +
    // MXINT8 (8-bit, E8M0). C=1024 is a multiple of 32, so every filter row's
    // bit-stream is word-aligned for all widths.
    conv2d_bench_fmt!(bench_int2, mt_int2_conv2d::kernel_ir_for, QFormat::Int2);
    conv2d_bench_fmt!(bench_int3, mt_int3_conv2d::kernel_ir_for, QFormat::Int3);
    conv2d_bench_fmt!(bench_int4, mt_int4_conv2d::kernel_ir_for, QFormat::Int4);
    conv2d_bench_fmt!(bench_int5, mt_int5_conv2d::kernel_ir_for, QFormat::Int5);
    conv2d_bench_fmt!(bench_int6, mt_int6_conv2d::kernel_ir_for, QFormat::Int6);
    conv2d_bench_fmt!(bench_mxint2, mt_mxint2_conv2d::kernel_ir_for, QFormat::Mxint2);
    conv2d_bench_fmt!(bench_mxint3, mt_mxint3_conv2d::kernel_ir_for, QFormat::Mxint3);
    conv2d_bench_fmt!(bench_mxint4, mt_mxint4_conv2d::kernel_ir_for, QFormat::Mxint4);
    conv2d_bench_fmt!(bench_mxint5, mt_mxint5_conv2d::kernel_ir_for, QFormat::Mxint5);
    conv2d_bench_fmt!(bench_mxint6, mt_mxint6_conv2d::kernel_ir_for, QFormat::Mxint6);
    conv2d_bench_fmt!(bench_mxint8, mt_mxint8_conv2d::kernel_ir_for, QFormat::Mxint8);
    // FP16-scale twins: same element packing as their FP32-scaled twin, scale
    // read as a native half. fp8_e4m3_f16 reuses the nvfp8_f16 kernel.
    conv2d_bench_fmt!(bench_nvfp8_f16, mt_nvfp8_f16_conv2d::kernel_ir_for, QFormat::Nvfp8F16);
    conv2d_bench_fmt!(bench_fp8_e4m3_f16, mt_nvfp8_f16_conv2d::kernel_ir_for, QFormat::Fp8E4m3F16);
    conv2d_bench_fmt!(bench_fp4_f16, mt_fp4_f16_conv2d::kernel_ir_for, QFormat::Fp4F16);
    conv2d_bench_fmt!(
        bench_fp8_e5m2_f16,
        mt_fp8_e5m2_f16_conv2d::kernel_ir_for,
        QFormat::Fp8E5m2F16
    );
    conv2d_bench_fmt!(bench_int2_f16, mt_int2_f16_conv2d::kernel_ir_for, QFormat::Int2F16);
    conv2d_bench_fmt!(bench_int3_f16, mt_int3_f16_conv2d::kernel_ir_for, QFormat::Int3F16);
    conv2d_bench_fmt!(bench_int4_f16, mt_int4_f16_conv2d::kernel_ir_for, QFormat::Int4F16);
    conv2d_bench_fmt!(bench_int5_f16, mt_int5_f16_conv2d::kernel_ir_for, QFormat::Int5F16);
    conv2d_bench_fmt!(bench_int6_f16, mt_int6_f16_conv2d::kernel_ir_for, QFormat::Int6F16);
    conv2d_bench_fmt!(bench_int8_f16, mt_int8_f16_conv2d::kernel_ir_for, QFormat::Int8F16);
}
