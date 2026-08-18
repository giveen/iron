//! Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//! SPDX-License-Identifier: Apache-2.0
//! Split a row-contiguous projection into one scattered output and two
//! contiguous outputs.
//!
//! The input layout is `[rows, scatter_width + 2 * copy_width]`. The first
//! segment overwrites indexed columns of a preinitialized primary output.
//! The second and third segments copy into independent row-contiguous outputs.

use wh_iron::kernel;

#[kernel]
pub fn iron_split_scatter_columns<T>(
    input: Tensor<T>,
    indices: Tensor<u32>,
    mut primary: Tensor<T>,
    mut secondary: Tensor<T>,
    mut tertiary: Tensor<T>,
    #[constexpr] scatter_width: u32,
    #[constexpr] primary_width: u32,
    #[constexpr] copy_width: u32,
    #[constexpr] n_elems: u32,
) {
    let idx = program_id::<0>();
    if idx < n_elems {
        let input_width = scatter_width + 2u32 * copy_width;
        let row = idx / input_width;
        let col = idx - row * input_width;
        let value = load(input[idx]);
        if col < scatter_width {
            let out_col = load(indices[col]);
            store(primary[row * primary_width + out_col], value);
        } else if col < scatter_width + copy_width {
            store(secondary[row * copy_width + col - scatter_width], value);
        } else {
            store(tertiary[row * copy_width + col - scatter_width - copy_width], value);
        }
    }
}

pub mod kernel_tests {
    use wh_iron::{test::*, test_kernel};

    use super::iron_split_scatter_columns;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn setup(rows: usize, dt: DType) -> TestSetup {
        let (scatter_width, primary_width, copy_width) = (3usize, 7usize, 2usize);
        let input_width = scatter_width + 2 * copy_width;
        let n_elems = rows * input_width;
        let input: Vec<f32> = (0..n_elems).map(|i| i as f32 * 0.125 - 2.0).collect();
        let input_dt = unpack_f32(&pack_f32(&input, dt), dt);
        let indices = vec![5u32, 1, 6];
        let primary_initial: Vec<f32> =
            (0..rows * primary_width).map(|i| 10.0 + i as f32 * 0.25).collect();
        let mut primary_expected = unpack_f32(&pack_f32(&primary_initial, dt), dt);
        let mut secondary_expected = vec![0.0f32; rows * copy_width];
        let mut tertiary_expected = vec![0.0f32; rows * copy_width];

        for row in 0..rows {
            let input_base = row * input_width;
            for col in 0..scatter_width {
                primary_expected[row * primary_width + indices[col] as usize] =
                    input_dt[input_base + col];
            }
            for col in 0..copy_width {
                secondary_expected[row * copy_width + col] =
                    input_dt[input_base + scatter_width + col];
                tertiary_expected[row * copy_width + col] =
                    input_dt[input_base + scatter_width + copy_width + col];
            }
        }

        TestSetup::new(iron_split_scatter_columns::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("input", pack_f32(&input, dt), dt))
            .input(TestBuffer::from_vec("indices", u32_bytes(&indices), DType::U32))
            .input(TestBuffer::from_vec("primary", pack_f32(&primary_initial, dt), dt))
            .input(TestBuffer::zeros("secondary", rows * copy_width, dt))
            .input(TestBuffer::zeros("tertiary", rows * copy_width, dt))
            .constexpr("scatter_width", scatter_width as u32)
            .constexpr("primary_width", primary_width as u32)
            .constexpr("copy_width", copy_width as u32)
            .constexpr("n_elems", n_elems as u32)
            .expect(TestBuffer::from_vec("primary", pack_f32(&primary_expected, dt), dt))
            .expect(TestBuffer::from_vec("secondary", pack_f32(&secondary_expected, dt), dt))
            .expect(TestBuffer::from_vec("tertiary", pack_f32(&tertiary_expected, dt), dt))
            .grid_1d(n_elems, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_split_scatter_columns_single_row(dt: DType) -> TestSetup { setup(1, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_split_scatter_columns_many_rows(dt: DType) -> TestSetup { setup(4, dt) }
}

pub mod kernel_benches {
    use wh_iron::{bench, test::*};

    use super::iron_split_scatter_columns;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_split_scatter_columns(dt: DType) -> BenchSetup {
        let (rows, scatter_width, primary_width, copy_width) =
            (8usize, 1024usize, 12_288usize, 1024usize);
        let input_width = scatter_width + 2 * copy_width;
        let n_elems = rows * input_width;
        let indices: Vec<u8> = (0..scatter_width)
            .flat_map(|index| ((index * 11 % primary_width) as u32).to_le_bytes())
            .collect();
        let values_moved = n_elems * 2;

        BenchSetup::new(iron_split_scatter_columns::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("input", n_elems, dt))
            .buffer(BenchBuffer::from_vec("indices", indices, DType::U32))
            .buffer(BenchBuffer::random("primary", rows * primary_width, dt).output())
            .buffer(BenchBuffer::zeros("secondary", rows * copy_width, dt).output())
            .buffer(BenchBuffer::zeros("tertiary", rows * copy_width, dt).output())
            .constexpr("scatter_width", scatter_width as u32)
            .constexpr("primary_width", primary_width as u32)
            .constexpr("copy_width", copy_width as u32)
            .constexpr("n_elems", n_elems as u32)
            .grid_1d(n_elems, 256)
            .bytes_moved((values_moved * dt.size_bytes()) as u64)
    }
}
