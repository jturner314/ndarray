#![allow(non_snake_case)]
#![feature(test)]

extern crate test;
use test::Bencher;

extern crate ndarray;
use ndarray::prelude::*;

extern crate rand;
use rand::{Rng, SeedableRng, StdRng};

const SEED: &[usize] = &[1, 2, 3];

macro_rules! bench_no_permute {
    ($name:ident, $dim:ty, $sorted_axes:expr, $shape:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut rng: StdRng = SeedableRng::from_seed(SEED);
            let sorted_axes = $sorted_axes;
            b.iter(|| {
                let arr = unsafe { Array::<u8, $dim>::uninitialized($shape) };
                let mut axes = sorted_axes.clone();
                rng.shuffle(&mut axes);
                arr
            })
        }
    };
    ($name:ident, $dim:ty, $sorted_axes:expr, $shape:expr,) => {
        bench_no_permute!($name, $dim, $sorted_axes, $shape);
    };
}

macro_rules! bench_permuted_axes {
    ($name:ident, $dim:ty, $sorted_axes:expr, $shape:expr, $method:ident) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut rng: StdRng = SeedableRng::from_seed(SEED);
            let sorted_axes = $sorted_axes;
            b.iter(|| {
                let arr = unsafe { Array::<u8, $dim>::uninitialized($shape) };
                let mut axes = sorted_axes.clone();
                rng.shuffle(&mut axes);
                arr.$method(axes)
            })
        }
    };
    ($name:ident, $dim:ty, $sorted_axes:expr, $shape:expr, $method:ident,) => {
        bench_permuted_axes!($name, $dim, $sorted_axes, $shape, $method);
    };
}

bench_no_permute!(
    no_permute_Ix3,
    Ix3,
    [0, 1, 2],
    (0, 0, 0),
);
bench_no_permute!(
    no_permute_Ix4,
    Ix4,
    [0, 1, 2, 3],
    (0, 0, 0, 0),
);
bench_no_permute!(
    no_permute_Ix5,
    Ix5,
    [0, 1, 2, 3, 4],
    (0, 0, 0, 0, 0),
);
bench_no_permute!(
    no_permute_Ix6,
    Ix6,
    [0, 1, 2, 3, 4, 5],
    (0, 0, 0, 0, 0, 0),
);
bench_no_permute!(
    no_permute_IxDyn3,
    IxDyn,
    vec![0, 1, 2],
    &[0, 0, 0][..],
);
bench_no_permute!(
    no_permute_IxDyn4,
    IxDyn,
    vec![0, 1, 2, 3],
    &[0, 0, 0, 0][..],
);
bench_no_permute!(
    no_permute_IxDyn5,
    IxDyn,
    vec![0, 1, 2, 3, 4],
    &[0, 0, 0, 0, 0][..],
);
bench_no_permute!(
    no_permute_IxDyn6,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5],
    &[0, 0, 0, 0, 0, 0][..],
);
bench_no_permute!(
    no_permute_IxDyn7,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5, 6],
    &[0, 0, 0, 0, 0, 0, 0][..],
);

// =============================================================================

bench_permuted_axes!(
    permuted_axes_impl_copy_Ix3,
    Ix3,
    [0, 1, 2],
    (0, 0, 0),
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_Ix4,
    Ix4,
    [0, 1, 2, 3],
    (0, 0, 0, 0),
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_Ix5,
    Ix5,
    [0, 1, 2, 3, 4],
    (0, 0, 0, 0, 0),
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_Ix6,
    Ix6,
    [0, 1, 2, 3, 4, 5],
    (0, 0, 0, 0, 0, 0),
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_IxDyn3,
    IxDyn,
    vec![0, 1, 2],
    &[0, 0, 0][..],
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_IxDyn4,
    IxDyn,
    vec![0, 1, 2, 3],
    &[0, 0, 0, 0][..],
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_IxDyn5,
    IxDyn,
    vec![0, 1, 2, 3, 4],
    &[0, 0, 0, 0, 0][..],
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_IxDyn6,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5],
    &[0, 0, 0, 0, 0, 0][..],
    permuted_axes,
);
bench_permuted_axes!(
    permuted_axes_impl_copy_IxDyn7,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5, 6],
    &[0, 0, 0, 0, 0, 0, 0][..],
    permuted_axes,
);

// =============================================================================

bench_permuted_axes!(
    permuted_axes_impl_inplace_Ix3,
    Ix3,
    [0, 1, 2],
    (0, 0, 0),
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_Ix4,
    Ix4,
    [0, 1, 2, 3],
    (0, 0, 0, 0),
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_Ix5,
    Ix5,
    [0, 1, 2, 3, 4],
    (0, 0, 0, 0, 0),
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_Ix6,
    Ix6,
    [0, 1, 2, 3, 4, 5],
    (0, 0, 0, 0, 0, 0),
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_IxDyn3,
    IxDyn,
    vec![0, 1, 2],
    &[0, 0, 0][..],
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_IxDyn4,
    IxDyn,
    vec![0, 1, 2, 3],
    &[0, 0, 0, 0][..],
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_IxDyn5,
    IxDyn,
    vec![0, 1, 2, 3, 4],
    &[0, 0, 0, 0, 0][..],
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_IxDyn6,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5],
    &[0, 0, 0, 0, 0, 0][..],
    permuted_axes_impl_inplace,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_IxDyn7,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5, 6],
    &[0, 0, 0, 0, 0, 0, 0][..],
    permuted_axes_impl_inplace,
);

// =============================================================================

bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_Ix3,
    Ix3,
    [0, 1, 2],
    (0, 0, 0),
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_Ix4,
    Ix4,
    [0, 1, 2, 3],
    (0, 0, 0, 0),
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_Ix5,
    Ix5,
    [0, 1, 2, 3, 4],
    (0, 0, 0, 0, 0),
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_Ix6,
    Ix6,
    [0, 1, 2, 3, 4, 5],
    (0, 0, 0, 0, 0, 0),
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_IxDyn3,
    IxDyn,
    vec![0, 1, 2],
    &[0, 0, 0][..],
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_IxDyn4,
    IxDyn,
    vec![0, 1, 2, 3],
    &[0, 0, 0, 0][..],
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_IxDyn5,
    IxDyn,
    vec![0, 1, 2, 3, 4],
    &[0, 0, 0, 0, 0][..],
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_IxDyn6,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5],
    &[0, 0, 0, 0, 0, 0][..],
    permuted_axes_impl_inplace_unchecked,
);
bench_permuted_axes!(
    permuted_axes_impl_inplace_unchecked_IxDyn7,
    IxDyn,
    vec![0, 1, 2, 3, 4, 5, 6],
    &[0, 0, 0, 0, 0, 0, 0][..],
    permuted_axes_impl_inplace_unchecked,
);
