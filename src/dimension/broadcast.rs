use error::*;
use std::cmp;
use std::iter;
use {Dimension, Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// This separate function contains most of the logic of
/// `BroadcastShapes::broadcast()` to minimize the size of the
/// monomorphized implementations.
fn broadcast<D: Dimension>(shape1: &[Ix], shape2: &[Ix]) -> Result<D, ShapeError> {
    // Zip the dims in reverse order, adding `&1`s to the shorter one until
    // they're the same length.
    let zipped = if shape1.len() < shape2.len() {
        shape1.iter()
            .rev()
            .chain(iter::repeat(&1))
            .zip(shape2.iter().rev())
    } else {
        shape2.iter()
            .rev()
            .chain(iter::repeat(&1))
            .zip(shape1.iter().rev())
    };
    let mut out = D::zero_index_with_ndim(cmp::max(shape1.len(), shape2.len()));
    for ((&len1, &len2), elem) in zipped.zip(out.slice_mut()) {
        if len1 == len2 {
            *elem = len1;
        } else if len1 == 1 {
            *elem = len2;
        } else if len2 == 1 {
            *elem = len1;
        } else {
            return Err(from_kind(ErrorKind::IncompatibleShape));
        }
    }
    Ok(out)
}

pub trait BroadcastShapes<Other: Dimension>: Dimension {
    /// The resulting dimension type after broadcasting.
    type Output: Dimension;
    /// Determines the shape after broadcasting the dimensions together.
    ///
    /// If the dimensions are not compatible, returns `Err`.
    ///
    /// Uses the [NumPy broadcasting rules]
    /// (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
    fn broadcast(&self, other: &Other) -> Result<<Self as BroadcastShapes<Other>>::Output, ShapeError> {
        broadcast(self.slice(), other.slice())
    }
}

impl<D: Dimension> BroadcastShapes<D> for D {
    type Output = D;
}

macro_rules! impl_broadcast_distinct_fixed {
    ($smaller:ty, $larger:ty) => {
        impl BroadcastShapes<$larger> for $smaller {
            type Output = $larger;
        }

        impl BroadcastShapes<$smaller> for $larger {
            type Output = $larger;
        }
    }
}

impl_broadcast_distinct_fixed!(Ix0, Ix1);
impl_broadcast_distinct_fixed!(Ix0, Ix2);
impl_broadcast_distinct_fixed!(Ix0, Ix3);
impl_broadcast_distinct_fixed!(Ix0, Ix4);
impl_broadcast_distinct_fixed!(Ix0, Ix5);
impl_broadcast_distinct_fixed!(Ix0, Ix6);
impl_broadcast_distinct_fixed!(Ix1, Ix2);
impl_broadcast_distinct_fixed!(Ix1, Ix3);
impl_broadcast_distinct_fixed!(Ix1, Ix4);
impl_broadcast_distinct_fixed!(Ix1, Ix5);
impl_broadcast_distinct_fixed!(Ix1, Ix6);
impl_broadcast_distinct_fixed!(Ix2, Ix3);
impl_broadcast_distinct_fixed!(Ix2, Ix4);
impl_broadcast_distinct_fixed!(Ix2, Ix5);
impl_broadcast_distinct_fixed!(Ix2, Ix6);
impl_broadcast_distinct_fixed!(Ix3, Ix4);
impl_broadcast_distinct_fixed!(Ix3, Ix5);
impl_broadcast_distinct_fixed!(Ix3, Ix6);
impl_broadcast_distinct_fixed!(Ix4, Ix5);
impl_broadcast_distinct_fixed!(Ix4, Ix6);
impl_broadcast_distinct_fixed!(Ix5, Ix6);

macro_rules! impl_broadcast_dyn {
    ($other:ty) => {
        impl BroadcastShapes<$other> for IxDyn {
            type Output = IxDyn;
        }

        impl BroadcastShapes<IxDyn> for $other {
            type Output = IxDyn;
        }
    }
}

impl_broadcast_dyn!(Ix0);
impl_broadcast_dyn!(Ix1);
impl_broadcast_dyn!(Ix2);
impl_broadcast_dyn!(Ix3);
impl_broadcast_dyn!(Ix4);
impl_broadcast_dyn!(Ix5);
impl_broadcast_dyn!(Ix6);
