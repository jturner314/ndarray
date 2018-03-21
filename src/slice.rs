// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use error::{ShapeError, ErrorKind};
use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeTo};
use std::fmt;
use std::marker::PhantomData;
use super::Dimension;

/// A slice (range with step size).
///
/// Negative `begin` or `end` indexes are counted from the back of the axis. If
/// `end` is `None`, the slice extends to the end of the axis.
///
/// See also the [`s![]`](macro.s.html) macro.
///
/// ## Examples
///
/// `Slice::new(0, None, 1)` is the full range of an axis. It can also be
/// created with `Slice::from(..)`. The Python equivalent is `[:]`.
///
/// `Slice::new(a, b, 2)` is every second element from `a` until `b`. It can
/// also be created with `Slice::from(a..b).step_by(2)`. The Python equivalent
/// is `[a:b:2]`.
///
/// `Slice::new(a, None, -1)` is every element, from `a` until the end, in
/// reverse order. It can also be created with `Slice::from(a..).step_by(-1)`.
/// The Python equivalent is `[a::-1]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Slice {
    pub start: isize,
    pub end: Option<isize>,
    pub step: isize,
}

impl Slice {
    /// Create a new `Slice` with the given extents.
    ///
    /// See also the `From` impls, converting from ranges; for example
    /// `Slice::from(i..)` or `Slice::from(j..k)`.
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Slice {
        debug_assert_ne!(step, 0, "Slice::new: step must be nonzero");
        Slice {
            start,
            end,
            step,
        }
    }

    /// Create a new `Slice` with the given step size (multiplied with the
    /// previous step size).
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    #[inline]
    pub fn step_by(self, step: isize) -> Self {
        debug_assert_ne!(step, 0, "Slice::step_by: step must be nonzero");
        Slice { step: self.step * step, ..self }
    }
}

macro_rules! impl_slice_from_index_type {
    ($index:ty) => {
        impl From<Range<$index>> for Slice {
            #[inline]
            fn from(r: Range<$index>) -> Slice {
                Slice {
                    start: r.start as isize,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }

        impl From<RangeFrom<$index>> for Slice {
            #[inline]
            fn from(r: RangeFrom<$index>) -> Slice {
                Slice {
                    start: r.start as isize,
                    end: None,
                    step: 1,
                }
            }
        }

        impl From<RangeTo<$index>> for Slice {
            #[inline]
            fn from(r: RangeTo<$index>) -> Slice {
                Slice {
                    start: 0,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }
    }
}

impl_slice_from_index_type!(isize);
impl_slice_from_index_type!(usize);
impl_slice_from_index_type!(i32);

impl From<RangeFull> for Slice {
    #[inline]
    fn from(_: RangeFull) -> Slice {
        Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

/// A slice (range with step) or an index.
///
/// See also the [`s![]`](macro.s!.html) macro for a convenient way to create a
/// `&SliceInfo<[SliceOrIndex; n], D>`.
///
/// ## Examples
///
/// `SliceOrIndex::Index(a)` is the index `a`. It can also be created with
/// `SliceOrIndex::from(a)`. The Python equivalent is `[a]`. The macro
/// equivalent is `s![a]`.
///
/// `SliceOrIndex::Slice { start: 0, end: None, step: 1 }` is the full range of
/// an axis. It can also be created with `SliceOrIndex::from(..)`. The Python
/// equivalent is `[:]`. The macro equivalent is `s![..]`.
///
/// `SliceOrIndex::Slice { start: a, end: Some(b), step: 2 }` is every second
/// element from `a` until `b`. It can also be created with
/// `SliceOrIndex::from(a..b).step_by(2)`. The Python equivalent is `[a:b:2]`.
/// The macro equivalent is `s![a..b;2]`.
///
/// `SliceOrIndex::Slice { start: a, end: None, step: -1 }` is every element,
/// from `a` until the end, in reverse order. It can also be created with
/// `SliceOrIndex::from(a..).step_by(-1)`. The Python equivalent is `[a::-1]`.
/// The macro equivalent is `s![a..;-1]`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SliceOrIndex {
    /// A range with step size. Negative `begin` or `end` indexes are counted
    /// from the back of the axis. If `end` is `None`, the slice extends to the
    /// end of the axis.
    Slice {
        start: isize,
        end: Option<isize>,
        step: isize,
    },
    /// A single index.
    Index(isize),
}

copy_and_clone!{SliceOrIndex}

impl SliceOrIndex {
    /// Returns `true` if `self` is a `Slice` value.
    pub fn is_slice(&self) -> bool {
        match self {
            &SliceOrIndex::Slice { .. } => true,
            _ => false,
        }
    }

    /// Returns `true` if `self` is an `Index` value.
    pub fn is_index(&self) -> bool {
        match self {
            &SliceOrIndex::Index(_) => true,
            _ => false,
        }
    }

    /// Returns a new `SliceOrIndex` with the given step size (multiplied with
    /// the previous step size).
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    #[inline]
    pub fn step_by(self, step: isize) -> Self {
        debug_assert_ne!(step, 0, "SliceOrIndex::step_by: step must be nonzero");
        match self {
            SliceOrIndex::Slice {
                start,
                end,
                step: orig_step,
            } => SliceOrIndex::Slice {
                start,
                end,
                step: orig_step * step,
            },
            SliceOrIndex::Index(s) => SliceOrIndex::Index(s),
        }
    }
}

impl fmt::Display for SliceOrIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SliceOrIndex::Index(index) => write!(f, "{}", index)?,
            SliceOrIndex::Slice { start, end, step } => {
                if start != 0 {
                    write!(f, "{}", start)?;
                }
                write!(f, "..")?;
                if let Some(i) = end {
                    write!(f, "{}", i)?;
                }
                if step != 1 {
                    write!(f, ";{}", step)?;
                }
            }
        }
        Ok(())
    }
}

impl From<Slice> for SliceOrIndex {
    #[inline]
    fn from(s: Slice) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: s.start,
            end: s.end,
            step: s.step,
        }
    }
}

macro_rules! impl_sliceorindex_from_index_type {
    ($index:ty) => {
        impl From<$index> for SliceOrIndex {
            #[inline]
            fn from(r: $index) -> SliceOrIndex {
                SliceOrIndex::Index(r as isize)
            }
        }

        impl From<Range<$index>> for SliceOrIndex {
            #[inline]
            fn from(r: Range<$index>) -> SliceOrIndex {
                SliceOrIndex::Slice {
                    start: r.start as isize,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }

        impl From<RangeFrom<$index>> for SliceOrIndex {
            #[inline]
            fn from(r: RangeFrom<$index>) -> SliceOrIndex {
                SliceOrIndex::Slice {
                    start: r.start as isize,
                    end: None,
                    step: 1,
                }
            }
        }

        impl From<RangeTo<$index>> for SliceOrIndex {
            #[inline]
            fn from(r: RangeTo<$index>) -> SliceOrIndex {
                SliceOrIndex::Slice {
                    start: 0,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }
    }
}

impl_sliceorindex_from_index_type!(isize);
impl_sliceorindex_from_index_type!(usize);
impl_sliceorindex_from_index_type!(i32);

impl From<RangeFull> for SliceOrIndex {
    #[inline]
    fn from(_: RangeFull) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

/// Represents all of the necessary information to perform a slice.
///
/// The type `T` is typically `[SliceOrIndex; n]`, `[SliceOrIndex]`, or
/// `Vec<SliceOrIndex>`. The type `D` is the output dimension after calling
/// [`.slice()`].
///
/// [`.slice()`]: struct.ArrayBase.html#method.slice
#[derive(Debug)]
#[repr(C)]
pub struct SliceInfo<T: ?Sized, D: Dimension> {
    out_dim: PhantomData<D>,
    indices: T,
}

impl<T: ?Sized, D> Deref for SliceInfo<T, D>
where
    D: Dimension,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

impl<T, D> SliceInfo<T, D>
where
    D: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// If you call this method, you are guaranteeing that `out_dim` is
    /// consistent with `indices`.
    #[doc(hidden)]
    pub unsafe fn new_unchecked(indices: T, out_dim: PhantomData<D>) -> SliceInfo<T, D> {
        SliceInfo {
            out_dim: out_dim,
            indices: indices,
        }
    }
}

impl<T, D> SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// Errors if `D` is not consistent with `indices`.
    pub fn new(indices: T) -> Result<SliceInfo<T, D>, ShapeError> {
        if let Some(ndim) = D::NDIM {
            if ndim != indices.as_ref().iter().filter(|s| s.is_slice()).count() {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        Ok(SliceInfo {
            out_dim: PhantomData,
            indices: indices,
        })
    }
}

impl<T: ?Sized, D> SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    /// Returns the number of dimensions after calling
    /// [`.slice()`](struct.ArrayBase.html#method.slice) (including taking
    /// subviews).
    ///
    /// If `D` is a fixed-size dimension type, then this is equivalent to
    /// `D::NDIM.unwrap()`. Otherwise, the value is calculated by iterating
    /// over the ranges/indices.
    pub fn out_ndim(&self) -> usize {
        D::NDIM.unwrap_or_else(|| {
            self.indices
                .as_ref()
                .iter()
                .filter(|s| s.is_slice())
                .count()
        })
    }
}

impl<T, D> AsRef<[SliceOrIndex]> for SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    fn as_ref(&self) -> &[SliceOrIndex] {
        self.indices.as_ref()
    }
}

impl<T, D> AsRef<SliceInfo<[SliceOrIndex], D>> for SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    fn as_ref(&self) -> &SliceInfo<[SliceOrIndex], D> {
        unsafe {
            // This is okay because the only non-zero-sized member of
            // `SliceInfo` is `indices`, so `&SliceInfo<[SliceOrIndex], D>`
            // should have the same bitwise representation as
            // `&[SliceOrIndex]`.
            &*(self.indices.as_ref() as *const [SliceOrIndex]
                as *const SliceInfo<[SliceOrIndex], D>)
        }
    }
}

impl<T, D> Copy for SliceInfo<T, D>
where
    T: Copy,
    D: Dimension,
{
}

impl<T, D> Clone for SliceInfo<T, D>
where
    T: Clone,
    D: Dimension,
{
    fn clone(&self) -> Self {
        SliceInfo {
            out_dim: PhantomData,
            indices: self.indices.clone(),
        }
    }
}


#[doc(hidden)]
pub trait SliceNextDim<D1, D2> {
    fn next_dim(&self, PhantomData<D1>) -> PhantomData<D2>;
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for Slice {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

macro_rules! impl_slicenextdim_for_index_type {
    ($index:ty) => {
        impl<D1: Dimension> SliceNextDim<D1, D1> for $index {
            fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1> {
                PhantomData
            }
        }
    }
}

impl_slicenextdim_for_index_type!(isize);
impl_slicenextdim_for_index_type!(usize);
impl_slicenextdim_for_index_type!(i32);

impl<D1: Dimension, T> SliceNextDim<D1, D1::Larger> for Range<T> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension, T> SliceNextDim<D1, D1::Larger> for RangeFrom<T> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension, T> SliceNextDim<D1, D1::Larger> for RangeTo<T> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for RangeFull {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices, separated by comma, with
/// optional step sizes that are separated from the range by a semicolon. It is
/// converted into a [`&SliceInfo`] instance.
///
/// [`&SliceInfo`]: struct.SliceInfo.html
///
/// Each range/slice/index uses signed indices, where a negative value is
/// counted from the end of the axis. Step sizes are also signed and may be
/// negative, but must not be zero.
///
/// The syntax is `s![` *[ axis-slice-or-index [, axis-slice-or-index [ , ... ]
/// ] ]* `]`, where *axis-slice-or-index* is any of the following:
///
/// * *index*: an index to use for taking a subview with respect to that axis
/// * *range*: a range with step size 1 to use for slicing that axis
/// * *range* `;` *step*: a range with step size *step* to use for slicing that axis
/// * *slice*: a [`Slice`] instance to use for slicing that axis
/// * *slice* `;` *step*: a range constructed from the start and end of a [`Slice`]
///   instance, with new step size *step*, to use for slicing that axis
///
/// [`Slice`]: struct.Slice.html
///
/// The number of *axis-slice-or-index* must match the number of axes in the
/// array. *index*, *range*, *slice*, and *step* can be expressions. *index*
/// must be of type `isize`, `usize`, or `i32`. *range* must be of type
/// `Range<I>`, `RangeTo<I>`, `RangeFrom<I>`, or `RangeFull` where `I` is
/// `isize`, `usize`, or `i32`. *step* must be a type that can be converted to
/// `isize` with the `as` keyword.
///
/// For example `s![0..4;2, 6, 1..5]` is a slice of the first axis for 0..4
/// with step size 2, a subview of the second axis at index 6, and a slice of
/// the third axis for 1..5 with default step size 1. The input array must have
/// 3 dimensions. The resulting slice would have shape `[2, 4]` for
/// [`.slice()`], [`.slice_mut()`], and [`.slice_move()`], and shape
/// `[2, 1, 4]` for [`.slice_inplace()`].
///
/// [`.slice()`]: struct.ArrayBase.html#method.slice
/// [`.slice_mut()`]: struct.ArrayBase.html#method.slice_mut
/// [`.slice_move()`]: struct.ArrayBase.html#method.slice_move
/// [`.slice_inplace()`]: struct.ArrayBase.html#method.slice_inplace
///
/// See also [*Slicing*](struct.ArrayBase.html#slicing).
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
///
/// use ndarray::{Array2, ArrayView2};
///
/// fn laplacian(v: &ArrayView2<f32>) -> Array2<f32> {
///     -4. * &v.slice(s![1..-1, 1..-1])
///     + v.slice(s![ ..-2, 1..-1])
///     + v.slice(s![1..-1,  ..-2])
///     + v.slice(s![1..-1, 2..  ])
///     + v.slice(s![2..  , 1..-1])
/// }
/// # fn main() { }
/// ```
///
/// # Negative *step*
///
/// The behavior of negative *step* arguments is most easily understood with
/// slicing as a two-step process:
///
/// 1. First, perform a slice with *range*.
///
/// 2. If *step* is positive, start with the front of the slice; if *step* is
///    negative, start with the back of the slice. Then, add *step* until
///    reaching the other end of the slice (inclusive).
///
/// An equivalent way to think about step 2 is, "If *step* is negative, reverse
/// the slice. Start at the front of the (possibly reversed) slice, and add
/// *step.abs()* until reaching the back of the slice (inclusive)."
///
/// For example,
///
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// #
/// # use ndarray::prelude::*;
/// #
/// # fn main() {
/// let arr = array![0, 1, 2, 3];
/// assert_eq!(arr.slice(s![1..3;-1]), array![2, 1]);
/// assert_eq!(arr.slice(s![1..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..4;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![..;-2]), array![3, 1]);
/// # }
/// ```
#[macro_export]
macro_rules! s(
    // convert a..b;c into @convert(a..b, c), final item
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        match $r {
            r => {
                let out_dim = $crate::SliceNextDim::next_dim(&r, $dim);
                #[allow(unsafe_code)]
                unsafe {
                    $crate::SliceInfo::new_unchecked(
                        [$($stack)* s!(@convert r, $s)],
                        out_dim,
                    )
                }
            }
        }
    };
    // convert a..b into @convert(a..b), final item
    (@parse $dim:expr, [$($stack:tt)*] $r:expr) => {
        match $r {
            r => {
                let out_dim = $crate::SliceNextDim::next_dim(&r, $dim);
                #[allow(unsafe_code)]
                unsafe {
                    $crate::SliceInfo::new_unchecked(
                        [$($stack)* s!(@convert r)],
                        out_dim,
                    )
                }
            }
        }
    };
    // convert a..b;c into @convert(a..b, c), final item, trailing comma
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        s![@parse $dim, [$($stack)*] $r;$s]
    };
    // convert a..b into @convert(a..b), final item, trailing comma
    (@parse $dim:expr, [$($stack:tt)*] $r:expr ,) => {
        s![@parse $dim, [$($stack)*] $r]
    };
    // convert a..b;c into @convert(a..b, c)
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        match $r {
            r => {
                s![@parse
                   $crate::SliceNextDim::next_dim(&r, $dim),
                   [$($stack)* s!(@convert r, $s),]
                   $($t)*
                ]
            }
        }
    };
    // convert a..b into @convert(a..b)
    (@parse $dim:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        match $r {
            r => {
                s![@parse
                   $crate::SliceNextDim::next_dim(&r, $dim),
                   [$($stack)* s!(@convert r),]
                   $($t)*
                ]
            }
        }
    };
    // convert range/index into SliceOrIndex
    (@convert $r:expr) => {
        <$crate::SliceOrIndex as ::std::convert::From<_>>::from($r)
    };
    // convert range/index and step into SliceOrIndex
    (@convert $r:expr, $s:expr) => {
        <$crate::SliceOrIndex as ::std::convert::From<_>>::from($r).step_by($s as isize)
    };
    ($($t:tt)*) => {
        // The extra `*&` is a workaround for this compiler bug:
        // https://github.com/rust-lang/rust/issues/23014
        &*&s![@parse ::std::marker::PhantomData::<$crate::Ix0>, [] $($t)*]
    };
);

/// Take multiple slices simultaneously.
///
/// This macro makes it possible to take multiple slices of the same array, as
/// long as Rust's aliasing rules are followed for *elements* in the slices.
/// For example, it's possible to take two disjoint, mutable slices of an
/// array, with one referencing the even-index elements and the other
/// referencing the odd-index elements. If you tried to achieve this by calling
/// `.slice_mut()` twice, the borrow checker would complain about mutably
/// borrowing the array twice (even though it's safe as long as the slices are
/// disjoint).
///
/// The syntax is `multislice!(` *expression, (pattern [, pattern [, …]])* `)`,
/// where *expression* evaluates to an `ArrayBase<S, D>` where `S: DataMut`,
/// and `pattern` is one of the following:
///
/// * `mut expr`: creates an `ArrayViewMut`, where `expr` evaluates to a
///   `&SliceInfo` instance used to slice the array.
/// * `expr`: creates an `ArrayView`, where `expr` evaluates to a `&SliceInfo`
///   instance used to slice the array.
///
/// **Note** that this macro always mutably borrows the array even if there are
/// no `mut` patterns. If all you want to do is take read-only slices, you
/// don't need `multislice!()`; just call
/// [`.slice()`](struct.ArrayBase.html#method.slice) multiple times instead.
///
/// `multislice!()` follows Rust's aliasing rules:
///
/// * An `ArrayViewMut` and `ArrayView` cannot reference the same element.
/// * Two `ArrayViewMut` cannot reference the same element.
/// * Two `ArrayView` can reference the same element.
///
/// **Panics** at runtime if any of the aliasing rules is violated.
///
/// See also [*Slicing*](struct.ArrayBase.html#slicing).
///
/// # Examples
///
/// In this example, there are two overlapping read-only slices, and two
/// disjoint mutable slices. Neither of the mutable slices intersects any of
/// the other slices.
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
///
/// use ndarray::prelude::*;
///
/// # fn main() {
/// let mut arr = Array1::from_iter(0..12);
/// let (a, b, c, d) = multislice!(arr, (s![0..5], mut s![6..;2], s![1..6], mut s![7..;2]));
/// assert_eq!(a, array![0, 1, 2, 3, 4]);
/// assert_eq!(b, array![6, 8, 10]);
/// assert_eq!(c, array![1, 2, 3, 4, 5]);
/// assert_eq!(d, array![7, 9, 11]);
/// # }
/// ```
///
/// These examples panic because they don't follow the aliasing rules:
///
/// * `ArrayViewMut` and `ArrayView` cannot reference the same element.
///
///   ```should_panic
///   # #[macro_use] extern crate ndarray;
///   # use ndarray::prelude::*;
///   # fn main() {
///   let mut arr = Array1::from_iter(0..12);
///   multislice!(arr, (s![0..5], mut s![1..;2])); // panic!
///   # }
///   ```
///
/// * Two `ArrayViewMut` cannot reference the same element.
///
///   ```should_panic
///   # #[macro_use] extern crate ndarray;
///   # use ndarray::prelude::*;
///   # fn main() {
///   let mut arr = Array1::from_iter(0..12);
///   multislice!(arr, (mut s![0..5], mut s![1..;2])); // panic!
///   # }
///   ```
#[macro_export]
macro_rules! multislice(
    (
        @check $view:expr,
        $info:expr,
        ()
    ) => {};
    // Check that $info doesn't intersect $other.
    (
        @check $view:expr,
        $info:expr,
        ($other:expr,)
    ) => {
        assert!(
            !$crate::slices_intersect(&$view.raw_dim(), $info, $other),
            "Slice {:?} must not intersect slice {:?}", $info, $other
        )
    };
    // Check that $info doesn't intersect any of the other info in the tuple.
    (
        @check $view:expr,
        $info:expr,
        ($other:expr, $($more:tt)*)
    ) => {
        {
            multislice!(@check $view, $info, ($other,));
            multislice!(@check $view, $info, ($($more)*));
        }
    };
    // Parse last slice (mutable), no trailing comma.
    (
        @parse $view:expr,
        ($($sliced:tt)*),
        ($($mut_info:tt)*),
        ($($immut_info:tt)*),
        (mut $info:expr)
    ) => {
        match $info {
            info => {
                multislice!(@check $view, info, ($($mut_info)*));
                multislice!(@check $view, info, ($($immut_info)*));
                ($($sliced)* unsafe { $view.aliasing_view_mut() }.slice_move(info))
            }
        }
    };
    // Parse last slice (read-only), no trailing comma.
    (
        @parse $view:expr,
        ($($sliced:tt)*),
        ($($mut_info:tt)*),
        ($($immut_info:tt)*),
        ($info:expr)
    ) => {
        match $info {
            info => {
                multislice!(@check $view, info, ($($mut_info)*));
                ($($sliced)* unsafe { $view.aliasing_view() }.slice_move(info))
            }
        }
    };
    // Parse last slice (mutable), with trailing comma.
    (
        @parse $view:expr,
        ($($sliced:tt)*),
        ($($mut_info:tt)*),
        ($($immut_info:tt)*),
        (mut $info:expr,)
    ) => {
        match $info {
            info => {
                multislice!(@check $view, info, ($($mut_info)*));
                multislice!(@check $view, info, ($($immut_info)*));
                ($($sliced)* unsafe { $view.aliasing_view_mut() }.slice_move(info))
            }
        }
    };
    // Parse last slice (read-only), with trailing comma.
    (
        @parse $view:expr,
        ($($sliced:tt)*),
        ($($mut_info:tt)*),
        ($($immut_info:tt)*),
        ($info:expr,)
    ) => {
        match $info {
            info => {
                multislice!(@check $view, info, ($($mut_info)*));
                ($($sliced)* unsafe { $view.aliasing_view() }.slice_move(info))
            }
        }
    };
    // Parse a mutable slice.
    (
        @parse $view:expr,
        ($($sliced:tt)*),
        ($($mut_info:tt)*),
        ($($immut_info:tt)*),
        (mut $info:expr, $($t:tt)*)
    ) => {
        match $info {
            info => {
                multislice!(@check $view, info, ($($mut_info)*));
                multislice!(@check $view, info, ($($immut_info)*));
                multislice!(
                    @parse $view,
                    ($($sliced)* unsafe { $view.aliasing_view_mut() }.slice_move(info),),
                    ($($mut_info)* info,),
                    ($($immut_info)*),
                    ($($t)*)
                )
            }
        }
    };
    // Parse a read-only slice.
    (
        @parse $view:expr,
        ($($sliced:tt)*),
        ($($mut_info:tt)*),
        ($($immut_info:tt)*),
        ($info:expr, $($t:tt)*)
    ) => {
        match $info {
            info => {
                multislice!(@check $view, info, ($($mut_info)*));
                multislice!(
                    @parse $view,
                    ($($sliced)* unsafe { $view.aliasing_view() }.slice_move(info),),
                    ($($mut_info)*),
                    ($($immut_info)* info,),
                    ($($t)*)
                )
            }
        }
    };
    // Entry point.
    ($arr:expr, ($($t:tt)*)) => {
        {
            let view = $crate::ArrayBase::view_mut(&mut $arr);
            multislice!(@parse view, (), (), (), ($($t)*))
        }
    };
);
