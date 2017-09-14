// Copyright 2014-2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use {Ix, Ix0, Ix1, IxDyn, Dimension, Dim, Axis};

/// Array shape with a next larger dimension.
///
/// `InsertAxis` defines a smaller-than relation for array shapes:
/// inserting one axis into *Self::Smaller* gives larger dimension *Self*.
pub trait InsertAxis : Dimension {
    fn insert_axis(&self_: &Self::Smaller, axis: Axis) -> Self;
}

// impl InsertAxis for Dim<[Ix; 6]> {
//     #[inline]
//     fn insert_axis(&self_: Self::Smaller, axis: Axis) -> IxDyn {
//         assert!(axis.index() <= self.ndim());
//         let vec = self.slice().to_vec();
//         vec.insert(axis.index(), 1);
//         vec.into()
//     }
// }

macro_rules! impl_insert_axis_array(
    ($($n:expr),*) => (
    $(
        impl InsertAxis for Dim<[Ix; $n]>
        {
            #[inline]
            fn insert_axis(&self_: &Self::Smaller, axis: Axis) -> Self {
                assert!(axis.index() <= self_.ndim());
                let mut tup = Dim([1; $n]);
                tup.slice_mut()[0..axis.index()].copy_from_slice(&self_.slice()[0..axis.index()]);
                tup.slice_mut()[axis.index()+1..$n].copy_from_slice(&self_.slice()[axis.index()..$n-1]);
                tup
            }
        }
    )*
    );
);

impl_insert_axis_array!(1, 2, 3, 4, 5, 6);
