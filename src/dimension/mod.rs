// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Ix, Ixs, Slice, SliceOrIndex};
use error::{from_kind, ErrorKind, ShapeError};
use num_integer::div_floor;

pub use self::dim::*;
pub use self::axis::Axis;
pub use self::conversion::IntoDimension;
pub use self::dimension_trait::Dimension;
pub use self::ndindex::NdIndex;
pub use self::remove_axis::RemoveAxis;
pub use self::axes::{axes_of, Axes, AxisDescription};
pub use self::dynindeximpl::IxDynImpl;

#[macro_use] mod macros;
mod axis;
mod conversion;
pub mod dim;
mod dimension_trait;
mod dynindeximpl;
mod ndindex;
mod remove_axis;
mod axes;

/// Calculate offset from `Ix` stride converting sign properly
#[inline(always)]
pub fn stride_offset(n: Ix, stride: Ix) -> isize {
    (n as isize) * ((stride as Ixs) as isize)
}

/// Check whether the given `dim` and `stride` lead to overlapping indices
///
/// There is overlap if, when iterating through the dimensions in the order
/// of maximum variation, the current stride is inferior to the sum of all
/// preceding strides multiplied by their corresponding dimensions.
///
/// The current implementation assumes strides to be positive
pub fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool {
    let order = strides._fastest_varying_stride_order();

    let dim = dim.slice();
    let strides = strides.slice();
    let mut prev_offset = 1;
    for &index in order.slice() {
        let d = dim[index];
        let s = strides[index];
        // any stride is ok if dimension is 1
        if d != 1 && (s as isize) < prev_offset {
            return true;
        }
        prev_offset = stride_offset(d, s);
    }
    false
}

/// Check whether the given dimension and strides are memory safe
/// to index the provided slice.
///
/// To be safe, no stride may be negative, and the offset corresponding
/// to the last element of each dimension should be smaller than the length
/// of the slice. Also, the strides should not allow a same element to be
/// referenced by two different index.
pub fn can_index_slice<A, D: Dimension>(data: &[A], dim: &D, strides: &D)
    -> Result<(), ShapeError>
{
    // check lengths of axes.
    let len = match dim.size_checked() {
        Some(l) => l,
        None => return Err(from_kind(ErrorKind::OutOfBounds)),
    };
    // check if strides are strictly positive (zero ok for len 0)
    for &s in strides.slice() {
        let s = s as Ixs;
        if s < 1 && (len != 0 || s < 0) {
            return Err(from_kind(ErrorKind::Unsupported));
        }
    }
    if len == 0 {
        return Ok(());
    }
    // check that the maximum index is in bounds
    let mut last_index = dim.clone();
    for index in last_index.slice_mut().iter_mut() {
        *index -= 1;
    }
    if let Some(offset) = stride_offset_checked_arithmetic(dim,
                                                           strides,
                                                           &last_index)
    {
        // offset is guaranteed to be positive so no issue converting
        // to usize here
        if (offset as usize) >= data.len() {
            return Err(from_kind(ErrorKind::OutOfBounds));
        }
        if dim_stride_overlap(dim, strides) {
            return Err(from_kind(ErrorKind::Unsupported));
        }
    } else {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    Ok(())
}

/// Return stride offset for this dimension and index.
///
/// Return None if the indices are out of bounds, or the calculation would wrap
/// around.
fn stride_offset_checked_arithmetic<D>(dim: &D, strides: &D, index: &D)
    -> Option<isize>
    where D: Dimension
{
    let mut offset = 0;
    for (&d, &i, &s) in izip!(dim.slice(), index.slice(), strides.slice()) {
        if i >= d {
            return None;
        }

        if let Some(offset_) = (i as isize)
                                   .checked_mul((s as Ixs) as isize)
                                   .and_then(|x| x.checked_add(offset)) {
            offset = offset_;
        } else {
            return None;
        }
    }
    Some(offset)
}

/// Stride offset checked general version (slices)
#[inline]
pub fn stride_offset_checked(dim: &[Ix], strides: &[Ix], index: &[Ix]) -> Option<isize> {
    if index.len() != dim.len() {
        return None;
    }
    let mut offset = 0;
    for (&d, &i, &s) in izip!(dim, index, strides) {
        if i >= d {
            return None;
        }
        offset += stride_offset(i, s);
    }
    Some(offset)
}

/// Implementation-specific extensions to `Dimension`
pub trait DimensionExt {
// note: many extensions go in the main trait if they need to be special-
// cased per dimension
    /// Get the dimension at `axis`.
    ///
    /// *Panics* if `axis` is out of bounds.
    #[inline]
    fn axis(&self, axis: Axis) -> Ix;

    /// Set the dimension at `axis`.
    ///
    /// *Panics* if `axis` is out of bounds.
    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix);
}

impl<D> DimensionExt for D
    where D: Dimension
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix {
        self[axis.index()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix) {
        self[axis.index()] = value;
    }
}

impl<'a> DimensionExt for [Ix]
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix {
        self[axis.index()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix) {
        self[axis.index()] = value;
    }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// **Panics** if `index` is larger than the size of the axis
// FIXME: Move to Dimension trait
pub fn do_sub<A, D: Dimension>(dims: &mut D, ptr: &mut *mut A, strides: &D,
                               axis: usize, index: Ix) {
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    ndassert!(index < dim,
              concat!("subview: Index {} must be less than axis length {} ",
                      "for array with shape {:?}"),
             index, dim, *dims);
    dims.slice_mut()[axis] = 1;
    let off = stride_offset(index, stride);
    unsafe {
        *ptr = ptr.offset(off);
    }
}

/// Compute the equivalent unsigned index given the axis length and signed index.
#[inline]
pub fn abs_index(len: Ix, index: Ixs) -> Ix {
    if index < 0 {
        len - (-index as Ix)
    } else {
        index as Ix
    }
}

/// Determines nonnegative start and end indices, and performs sanity checks.
///
/// The return value is (start, end, step).
///
/// **Panics** if stride is 0 or if any index is out of bounds.
fn to_abs_slice(axis_len: usize, slice: Slice) -> (usize, usize, isize) {
    let Slice { start, end, step } = slice;
    let start = abs_index(axis_len, start);
    let mut end = abs_index(axis_len, end.unwrap_or(axis_len as isize));
    if end < start {
        end = start;
    }
    ndassert!(
        start <= axis_len,
        "Slice begin {} is past end of axis of length {}",
        start,
        axis_len,
    );
    ndassert!(
        end <= axis_len,
        "Slice end {} is past end of axis of length {}",
        end,
        axis_len,
    );
    ndassert!(step != 0, "Slice stride must not be zero");
    (start, end, step)
}

/// Modify dimension, stride and return data pointer offset
///
/// **Panics** if stride is 0 or if any index is out of bounds.
pub fn do_slice(dim: &mut usize, stride: &mut usize, slice: Slice) -> isize {
    let (start, end, step) = to_abs_slice(*dim, slice);

    let m = end - start;
    let s = (*stride) as isize;

    // Data pointer offset
    let mut offset = stride_offset(start, *stride);
    // Adjust for strides
    //
    // How to implement negative strides:
    //
    // Increase start pointer by
    // old stride * (old dim - 1)
    // to put the pointer completely in the other end
    if step < 0 {
        offset += stride_offset(m - 1, *stride);
    }

    let s_prim = s * step;

    let d = m / step.abs() as usize;
    let r = m % step.abs() as usize;
    let m_prim = d + if r > 0 { 1 } else { 0 };

    // Update dimension and stride coordinate
    *dim = m_prim;
    *stride = s_prim as usize;

    offset
}

/// Solves `a * x + b * y = gcd(a, b)` for `x`, `y`, and `gcd(a, b)`.
///
/// Returns `(g, (x, y))`, where `g` is `gcd(a, b)`, and `g` is always
/// nonnegative.
///
/// See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
fn extended_gcd(a: isize, b: isize) -> (isize, (isize, isize)) {
    if a == 0 {
        (b.abs(), (0, b.signum()))
    } else if b == 0 {
        (a.abs(), (a.signum(), 0))
    } else {
        let mut r = (a, b);
        let mut s = (1, 0);
        let mut t = (0, 1);
        while r.1 != 0 {
            let q = r.0 / r.1;
            r = (r.1, r.0 - q * r.1);
            s = (s.1, s.0 - q * s.1);
            t = (t.1, t.0 - q * t.1);
        }
        if r.0 > 0 {
            (r.0, (s.0, t.0))
        } else {
            (-r.0, (-s.0, -t.0))
        }
    }
}

/// Solves `a * x + b * y = c` for `x` where `a`, `b`, `c`, `x`, and `y` are
/// integers.
///
/// If the return value is `Some((x0, xd))`, there is a solution. `xd` is
/// always positive. Solutions `x` are given by `x0 + xd * t` where `t` is any
/// integer. The value of `y` for any `x` is then `y = (c - a * x) / b`.
///
/// If the return value is `None`, no solutions exist.
///
/// **Note** `a` and `b` must be nonzero.
///
/// See https://en.wikipedia.org/wiki/Diophantine_equation#One_equation
/// and https://math.stackexchange.com/questions/1656120#1656138
fn solve_linear_diophantine_eq(a: isize, b: isize, c: isize) -> Option<(isize, isize)> {
    debug_assert_ne!(a, 0);
    debug_assert_ne!(b, 0);
    let (g, (u, _)) = extended_gcd(a, b);
    if c % g == 0 {
        Some((c / g * u, (b / g).abs()))
    } else {
        None
    }
}

/// Returns `true` if two (finite length) arithmetic sequences intersect.
///
/// `min*` and `max*` are the (inclusive) bounds of the sequences, and they
/// must be elements in the sequences. `step*` are the steps between
/// consecutive elements (the sign is irrelevant).
///
/// **Note** `step1` and `step2` must be nonzero.
fn arith_seq_intersect(
    (min1, max1, step1): (isize, isize, isize),
    (min2, max2, step2): (isize, isize, isize),
) -> bool {
    debug_assert!(max1 >= min1);
    debug_assert!(max2 >= min2);
    debug_assert_eq!((max1 - min1) % step1, 0);
    debug_assert_eq!((max2 - min2) % step2, 0);

    // Handle the easy case where we don't have to solve anything.
    if min1 > max2 || min2 > max1 {
        false
    } else {
        // The sign doesn't matter semantically, and it's mathematically convenient
        // for `step1` and `step2` to be positive.
        let step1 = step1.abs();
        let step2 = step2.abs();
        // Ignoring the min/max bounds, the sequences are
        //   a(x) = min1 + step1 * x
        //   b(y) = min2 + step2 * y
        //
        // For intersections a(x) = b(y), we have:
        //   min1 + step1 * x = min2 + step2 * y
        //   ⇒ -step1 * x + step2 * y = min1 - min2
        // which is a linear Diophantine equation.
        if let Some((x0, xd)) = solve_linear_diophantine_eq(-step1, step2, min1 - min2) {
            // Minimum of [min1, max1] ∩ [min2, max2]
            let min = ::std::cmp::max(min1, min2);
            // Maximum of [min1, max1] ∩ [min2, max2]
            let max = ::std::cmp::min(max1, max2);
            // The potential intersections are
            //   a(x) = min1 + step1 * (x0 + xd * t)
            // where `t` is any integer.
            //
            // There is an intersection in `[min, max]` if there exists an
            // integer `t` such that
            //   min ≤ a(x) ≤ max
            //   ⇒ min ≤ min1 + step1 * (x0 + xd * t) ≤ max
            //   ⇒ min ≤ min1 + step1 * x0 + step1 * xd * t ≤ max
            //   ⇒ min - min1 - step1 * x0 ≤ (step1 * xd) * t ≤ max - min1 - step1 * x0
            //
            // Therefore, the least possible intersection `a(x)` that is ≥ `min` has
            //   t = ⌈(min - min1 - step1 * x0) / (step1 * xd)⌉
            // If this `a(x) is also ≤ `max`, then there is an intersection in `[min, max]`.
            //
            // The greatest possible intersection `a(x)` that is ≤ `max` has
            //   t = ⌊(max - min1 - step1 * x0) / (step1 * xd)⌋
            // If this `a(x) is also ≥ `min`, then there is an intersection in `[min, max]`.
            min1 + step1 * (x0 - xd * div_floor(min - min1 - step1 * x0, -step1 * xd)) <= max
                || min1 + step1 * (x0 + xd * div_floor(max - min1 - step1 * x0, step1 * xd)) >= min
        } else {
            false
        }
    }
}

/// Returns the minimum and maximum values of the indices (inclusive).
///
/// If the slice is empty, then returns `None`, otherwise returns `Some((min, max))`.
fn slice_min_max(axis_len: usize, slice: Slice) -> Option<(usize, usize)> {
    let (start, end, step) = to_abs_slice(axis_len, slice);
    if start == end {
        None
    } else {
        if step > 0 {
            Some((start, end - 1 - (end - start - 1) % (step as usize)))
        } else {
            Some((start + (end - start - 1) % (-step as usize), end - 1))
        }
    }
}

/// Returns `true` iff the slices intersect.
#[doc(hidden)]
pub fn slices_intersect<D: Dimension>(
    dim: &D,
    indices1: &D::SliceArg,
    indices2: &D::SliceArg,
) -> bool {
    debug_assert_eq!(indices1.as_ref().len(), indices2.as_ref().len());
    for (&axis_len, &si1, &si2) in izip!(dim.slice(), indices1.as_ref(), indices2.as_ref()) {
        // The slices do not intersect iff any pair of `SliceOrIndex` does not intersect.
        match (si1, si2) {
            (
                SliceOrIndex::Slice {
                    start: start1,
                    end: end1,
                    step: step1,
                },
                SliceOrIndex::Slice {
                    start: start2,
                    end: end2,
                    step: step2,
                },
            ) => {
                let (min1, max1) = match slice_min_max(axis_len, Slice::new(start1, end1, step1)) {
                    Some(m) => m,
                    None => return false,
                };
                let (min2, max2) = match slice_min_max(axis_len, Slice::new(start2, end2, step2)) {
                    Some(m) => m,
                    None => return false,
                };
                if !arith_seq_intersect(
                    (min1 as isize, max1 as isize, step1),
                    (min2 as isize, max2 as isize, step2),
                ) {
                    return false;
                }
            }
            (SliceOrIndex::Slice { start, end, step }, SliceOrIndex::Index(ind)) |
            (SliceOrIndex::Index(ind), SliceOrIndex::Slice { start, end, step }) => {
                let ind = abs_index(axis_len, ind);
                let (min, max) = match slice_min_max(axis_len, Slice::new(start, end, step)) {
                    Some(m) => m,
                    None => return false,
                };
                if ind < min || ind > max || (ind - min) % step.abs() as usize != 0 {
                    return false;
                }
            }
            (SliceOrIndex::Index(ind1), SliceOrIndex::Index(ind2)) => {
                let ind1 = abs_index(axis_len, ind1);
                let ind2 = abs_index(axis_len, ind2);
                if ind1 != ind2 {
                    return false;
                }
            }
        }
    }
    true
}

pub fn merge_axes<D>(dim: &mut D, strides: &mut D, take: Axis, into: Axis) -> bool
    where D: Dimension,
{
    let il = dim.axis(into);
    let is = strides.axis(into) as Ixs;
    let tl = dim.axis(take);
    let ts = strides.axis(take) as Ixs;
    if il as Ixs * is != ts {
        return false;
    }
    // merge them
    dim.set_axis(into, il * tl);
    dim.set_axis(take, 1);
    true
}


// NOTE: These tests are not compiled & tested
#[cfg(test)]
mod test {
    use super::{arith_seq_intersect, extended_gcd, slice_min_max, slices_intersect,
                solve_linear_diophantine_eq, IntoDimension};
    use Dim;
    use error::{from_kind, ErrorKind};
    use num_integer::gcd;
    use quickcheck::TestResult;
    use slice::Slice;

    #[test]
    fn slice_indexing_uncommon_strides() {
        let v: Vec<_> = (0..12).collect();
        let dim = (2, 3, 2).into_dimension();
        let strides = (1, 2, 6).into_dimension();
        assert!(super::can_index_slice(&v, &dim, &strides).is_ok());

        let strides = (2, 4, 12).into_dimension();
        assert_eq!(super::can_index_slice(&v, &dim, &strides),
                   Err(from_kind(ErrorKind::OutOfBounds)));
    }

    #[test]
    fn overlapping_strides_dim() {
        let dim = (2, 3, 2).into_dimension();
        let strides = (5, 2, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 2, 1).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 0, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
    }

    quickcheck! {
        fn extended_gcd_solves_eq(a: isize, b: isize) -> bool {
            let (g, (x, y)) = extended_gcd(a, b);
            a * x + b * y == g
        }

        fn extended_gcd_correct_gcd(a: isize, b: isize) -> bool {
            let (g, _) = extended_gcd(a, b);
            g == gcd(a, b)
        }
    }

    #[test]
    fn extended_gcd_zero() {
        assert_eq!(extended_gcd(0, 0), (0, (0, 0)));
        assert_eq!(extended_gcd(0, 5), (5, (0, 1)));
        assert_eq!(extended_gcd(5, 0), (5, (1, 0)));
        assert_eq!(extended_gcd(0, -5), (5, (0, -1)));
        assert_eq!(extended_gcd(-5, 0), (5, (-1, 0)));
    }

    quickcheck! {
        fn solve_linear_diophantine_eq_solution_existence(
            a: isize, b: isize, c: isize
        ) -> TestResult {
            if a == 0 || b == 0 {
                TestResult::discard()
            } else {
                TestResult::from_bool(
                    (c % gcd(a, b) == 0) == solve_linear_diophantine_eq(a, b, c).is_some()
                )
            }
        }

        fn solve_linear_diophantine_eq_correct_solution(
            a: isize, b: isize, c: isize, t: isize
        ) -> TestResult {
            if a == 0 || b == 0 {
                TestResult::discard()
            } else {
                match solve_linear_diophantine_eq(a, b, c) {
                    Some((x0, xd)) => {
                        let x = x0 + xd * t;
                        let y = (c - a * x) / b;
                        TestResult::from_bool(a * x + b * y == c)
                    }
                    None => TestResult::discard(),
                }
            }
        }
    }

    quickcheck! {
        fn arith_seq_intersect_correct(
            first1: isize, len1: isize, step1: isize,
            first2: isize, len2: isize, step2: isize
        ) -> TestResult {
            use std::cmp;

            if len1 == 0 || len2 == 0 {
                // This case is impossible to reach in `arith_seq_intersect()`
                // because the `min*` and `max*` arguments are inclusive.
                return TestResult::discard();
            }
            let len1 = len1.abs();
            let len2 = len2.abs();

            // Convert to `min*` and `max*` arguments for `arith_seq_intersect()`.
            let last1 = first1 + step1 * (len1 - 1);
            let (min1, max1) = (cmp::min(first1, last1), cmp::max(first1, last1));
            let last2 = first2 + step2 * (len2 - 1);
            let (min2, max2) = (cmp::min(first2, last2), cmp::max(first2, last2));

            // Naively determine if the sequences intersect.
            let seq1: Vec<_> = (0..len1)
                .map(|n| first1 + step1 * n)
                .collect();
            let intersects = (0..len2)
                .map(|n| first2 + step2 * n)
                .any(|elem2| seq1.contains(&elem2));

            TestResult::from_bool(
                arith_seq_intersect(
                    (min1, max1, if step1 == 0 { 1 } else { step1 }),
                    (min2, max2, if step2 == 0 { 1 } else { step2 })
                ) == intersects
            )
        }
    }

    #[test]
    fn slice_min_max_empty() {
        assert_eq!(slice_min_max(0, Slice::new(0, None, 3)), None);
        assert_eq!(slice_min_max(10, Slice::new(1, Some(1), 3)), None);
        assert_eq!(slice_min_max(10, Slice::new(-1, Some(-1), 3)), None);
        assert_eq!(slice_min_max(10, Slice::new(1, Some(1), -3)), None);
        assert_eq!(slice_min_max(10, Slice::new(-1, Some(-1), -3)), None);
    }

    #[test]
    fn slice_min_max_pos_step() {
        assert_eq!(slice_min_max(10, Slice::new(1, Some(8), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(9), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(8), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(9), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(-2), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(-1), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(-2), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(-1), 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, None, 3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, None, 3)), Some((1, 7)));
        assert_eq!(slice_min_max(11, Slice::new(1, None, 3)), Some((1, 10)));
        assert_eq!(slice_min_max(11, Slice::new(-10, None, 3)), Some((1, 10)));
    }

    #[test]
    fn slice_min_max_neg_step() {
        assert_eq!(slice_min_max(10, Slice::new(1, Some(8), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(2, Some(8), -3)), Some((4, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(8), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-8, Some(8), -3)), Some((4, 7)));
        assert_eq!(slice_min_max(10, Slice::new(1, Some(-2), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(2, Some(-2), -3)), Some((4, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-9, Some(-2), -3)), Some((1, 7)));
        assert_eq!(slice_min_max(10, Slice::new(-8, Some(-2), -3)), Some((4, 7)));
        assert_eq!(slice_min_max(9, Slice::new(2, None, -3)), Some((2, 8)));
        assert_eq!(slice_min_max(9, Slice::new(-7, None, -3)), Some((2, 8)));
        assert_eq!(slice_min_max(9, Slice::new(3, None, -3)), Some((5, 8)));
        assert_eq!(slice_min_max(9, Slice::new(-6, None, -3)), Some((5, 8)));
    }

    #[test]
    fn slices_intersect_true() {
        assert!(slices_intersect(&Dim([4, 5]), s![.., ..], s![.., ..]));
        assert!(slices_intersect(&Dim([4, 5]), s![0, ..], s![0, ..]));
        assert!(slices_intersect(&Dim([4, 5]), s![..;2, ..], s![..;3, ..]));
        assert!(slices_intersect(&Dim([4, 5]), s![.., ..;2], s![.., 1..;3]));
        assert!(slices_intersect(&Dim([4, 10]), s![.., ..;9], s![.., 3..;6]));
    }

    #[test]
    fn slices_intersect_false() {
        assert!(!slices_intersect(&Dim([4, 5]), s![..;2, ..], s![1..;2, ..]));
        assert!(!slices_intersect(&Dim([4, 5]), s![..;2, ..], s![1..;3, ..]));
        assert!(!slices_intersect(&Dim([4, 5]), s![.., ..;9], s![.., 3..;6]));
    }
}
