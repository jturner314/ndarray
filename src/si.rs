// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::fmt;
use std::marker::PhantomData;
use super::{Dimension, Ixs};

// [a:b:s] syntax for example [:3], [::-1]
// [0,:] -- first row of matrix
// [:,0] -- first column of matrix

#[derive(PartialEq, Eq, Hash)]
/// A slice, a description of a range of an array axis.
///
/// Fields are `begin`, `end` and `stride`, where
/// negative `begin` or `end` indexes are counted from the back
/// of the axis.
///
/// If `end` is `None`, the slice extends to the end of the axis.
///
/// See also the [`s![] macro`](macro.s!.html), a convenient way to specify
/// an array of `Si`.
///
/// ## Examples
///
/// `Si(0, None, 1)` is the full range of an axis.
/// Python equivalent is `[:]`. Macro equivalent is `s![..]`.
///
/// `Si(a, Some(b), 2)` is every second element from `a` until `b`.
/// Python equivalent is `[a:b:2]`. Macro equivalent is `s![a..b;2]`.
///
/// `Si(a, None, -1)` is every element, from `a`
/// until the end, in reverse order. Python equivalent is `[a::-1]`.
/// Macro equivalent is `s![a..;-1]`.
///
/// The constant [`S`] is a shorthand for the full range of an axis.
/// [`S`]: constant.S.html
pub struct Si(pub Ixs, pub Option<Ixs>, pub Ixs);

impl fmt::Debug for Si {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Si(0, _, _) => { }
            Si(i, _, _) => { try!(write!(f, "{}", i)); }
        }
        try!(write!(f, ".."));
        match *self {
            Si(_, None, _) => { }
            Si(_, Some(i), _) => { try!(write!(f, "{}", i)); }
        }
        match *self {
            Si(_, _, 1) => { }
            Si(_, _, s) => { try!(write!(f, ";{}", s)); }
        }
        Ok(())
    }
}

impl From<Range<Ixs>> for Si {
    #[inline]
    fn from(r: Range<Ixs>) -> Si {
        Si(r.start, Some(r.end), 1)
    }
}

impl From<RangeFrom<Ixs>> for Si {
    #[inline]
    fn from(r: RangeFrom<Ixs>) -> Si {
        Si(r.start, None, 1)
    }
}

impl From<RangeTo<Ixs>> for Si {
    #[inline]
    fn from(r: RangeTo<Ixs>) -> Si {
        Si(0, Some(r.end), 1)
    }
}

impl From<RangeFull> for Si {
    #[inline]
    fn from(_: RangeFull) -> Si {
        S
    }
}


impl Si {
    #[inline]
    pub fn step(self, step: Ixs) -> Self {
        Si(self.0, self.1, self.2 * step)
    }
}

copy_and_clone!{Si}

/// Slice value for the full range of an axis.
pub const S: Si = Si(0, None, 1);

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges, separated by comma, with optional strides
/// that are separated from the range by a semicolon.
/// It is converted into a slice argument with type `&[Si; N]`.
///
/// Each range uses signed indices, where a negative value is counted from
/// the end of the axis. Strides are also signed and may be negative, but
/// must not be zero.
///
/// The syntax is `s![` *[ axis-slice [, axis-slice [ , ... ] ] ]* `]`.
/// Where *axis-slice* is either *i* `..` *j* or *i* `..` *j* `;` *step*,
/// and *i* is the start index, *j* end index and *step* the element step
/// size (which defaults to 1). The number of *axis-slice* must match the
/// number of axes in the array.
///
/// For example `s![0..4;2, 1..5]` is a slice of rows 0..4 with step size 2,
/// and columns 1..5 with default step size 1. The slice would have
/// shape `[2, 4]`.
///
/// If an array has two axes, the slice argument is passed as
/// type `&[Si; 2]`.  The macro expansion of `s![a..b;c, d..e]`
/// is equivalent to `&[Si(a, Some(b), c), Si(d, Some(e), 1)]`.
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
#[macro_export]
macro_rules! s(
    // convert a..b;c into @step(a..b, c), final item
    (@parse [$($stack:tt)*] $r:expr;$s:expr) => {
        &[$($stack)* s!(@step $r, $s)]
    };
    // convert a..b into @step(a..b, 1), final item
    (@parse [$($stack:tt)*] $r:expr) => {
        &[$($stack)* s!(@step $r, 1)]
    };
    // convert a..b;c into @step(a..b, c), final item, trailing comma
    (@parse [$($stack:tt)*] $r:expr;$s:expr ,) => {
        &[$($stack)* s!(@step $r, $s)]
    };
    // convert a..b into @step(a..b, 1), final item, trailing comma
    (@parse [$($stack:tt)*] $r:expr ,) => {
        &[$($stack)* s!(@step $r, 1)]
    };
    // convert a..b;c into @step(a..b, c)
    (@parse [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        s![@parse [$($stack)* s!(@step $r, $s),] $($t)*]
    };
    // convert a..b into @step(a..b, 1)
    (@parse [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s![@parse [$($stack)* s!(@step $r, 1),] $($t)*]
    };
    // convert range, step into Si
    (@step $r:expr, $s:expr) => {
        <$crate::Si as ::std::convert::From<_>>::from($r).step($s)
    };
    ($($t:tt)*) => {
        s![@parse [] $($t)*]
    };
);

pub struct SliceIndex<T: ?Sized + AsRef<[Si2]>, D: Dimension> {
    dim: PhantomData<D>,
    ind: T,
}

impl<T: ?Sized + AsRef<[Si2]>, D: Dimension> SliceIndex<T, D> {
    pub fn indices(&self) -> &[Si2] {
        self.ind.as_ref()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Si2 {
    Range(Si),
    Subview(isize),
}

impl Si2 {
    #[inline]
    pub fn step(self, step: Ixs) -> Self {
        match self {
            Si2::Range(s) => Si2::Range(s.step(step)),
            Si2::Subview(s) => Si2::Subview(s),
        }
    }
}

impl From<Range<Ixs>> for Si2 {
    #[inline]
    fn from(r: Range<Ixs>) -> Si2 {
        Si2::Range(Si::from(r))
    }
}

impl From<isize> for Si2 {
    #[inline]
    fn from(r: isize) -> Si2 {
        Si2::Subview(r)
    }
}

impl From<RangeFrom<Ixs>> for Si2 {
    #[inline]
    fn from(r: RangeFrom<Ixs>) -> Si2 {
        Si2::Range(Si::from(r))
    }
}

impl From<RangeTo<Ixs>> for Si2 {
    #[inline]
    fn from(r: RangeTo<Ixs>) -> Si2 {
        Si2::Range(Si::from(r))
    }
}

impl From<RangeFull> for Si2 {
    #[inline]
    fn from(r: RangeFull) -> Si2 {
        Si2::Range(Si::from(r))
    }
}

pub trait NewDim<D1, D2> {
    fn new_dim(&self, PhantomData<D1>) -> PhantomData<D2>;
}

impl<D1: Dimension> NewDim<D1, D1::Larger> for Range<Ixs> {
    fn new_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> NewDim<D1, D1::Larger> for RangeFrom<Ixs> {
    fn new_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> NewDim<D1, D1::Larger> for RangeTo<Ixs> {
    fn new_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> NewDim<D1, D1::Larger> for RangeFull {
    fn new_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> NewDim<D1, D1> for isize {
    fn new_dim(&self, _: PhantomData<D1>) -> PhantomData<D1> {
        PhantomData
    }
}

#[macro_export]
macro_rules! s2(
    // convert a..b;c into @step(a..b, c), final item
    (@parse $d:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        SliceIndex {
            dim: $crate::NewDim::new_dim(&$r, $d),
            ind: &[$($stack)* s2!(@step $r, $s)],
        }
    };
    // convert a..b into @step(a..b, 1), final item
    (@parse $d:expr, [$($stack:tt)*] $r:expr) => {
        SliceIndex {
            dim: $crate::NewDim::new_dim(&$r, $d),
            ind: &[$($stack)* s2!(@step $r, 1)],
        }
    };
    // convert a..b;c into @step(a..b, c), final item, trailing comma
    (@parse $d:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        SliceIndex {
            dim: $crate::NewDim::new_dim(&$r, $d),
            ind: &[$($stack)* s2!(@step $r, $s)],
        }
    };
    // convert a..b into @step(a..b, 1), final item, trailing comma
    (@parse $d:expr, [$($stack:tt)*] $r:expr ,) => {
        SliceIndex {
            dim: $crate::NewDim::new_dim(&$r, $d),
            ind: &[$($stack)* s2!(@step $r, 1)],
        }
    };
    // convert a..b;c into @step(a..b, c)
    (@parse $d:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        s2![@parse $r.new_dim($d), [$($stack)* s2!(@step $r, $s),] $($t)*]
    };
    // convert a..b into @step(a..b, 1)
    (@parse $d:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s2![@parse $crate::NewDim::new_dim(&$r, $d), [$($stack)* s2!(@step $r, 1),] $($t)*]
    };
    // convert range, step into Si2
    (@step $r:expr, $s:expr) => {
        <$crate::Si2 as ::std::convert::From<_>>::from($r).step($s)
    };
    ($($t:tt)*) => {
        s2![@parse ::std::marker::PhantomData::<$crate::Ix0>, [] $($t)*]
    };
);

#[cfg(test)]
mod tests {
    use super::{SliceIndex, Si, Si2};
    use super::super::Ix2;

    #[test]
    fn s2() {
        let indexes: SliceIndex<&[Si2; 3], Ix2> = s2![1..5, 6, 4..;2];
        assert_eq!(
            indexes.ind,
            &[
                Si2::Range(Si(1, Some(5), 1)),
                Si2::Subview(6),
                Si2::Range(Si(4, None, 2))
            ]
        );
        let _ = s2![1..5, 6, ..4;2];
    }
}
