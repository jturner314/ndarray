use error::*;
use std::cmp;
use std::iter;
use {Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

fn cobroadcast_pair<D1, D2, D3>(dim1: &D1, dim2: &D2) -> Result<D3, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    // Zip the dims in reverse order, adding `&1`s to the shorter one until
    // they're the same length.
    let zipped = if dim1.ndim() < dim2.ndim() {
        dim1.slice()
            .iter()
            .rev()
            .chain(iter::repeat(&1))
            .zip(dim2.slice().iter().rev())
    } else {
        dim2.slice()
            .iter()
            .rev()
            .chain(iter::repeat(&1))
            .zip(dim1.slice().iter().rev())
    };
    let mut out = D3::zero_index_with_ndim(cmp::max(dim1.ndim(), dim2.ndim()));
    for (i, (&len1, &len2)) in zipped.enumerate() {
        if len1 == len2 {
            out[i] = len1;
        } else if len1 == 1 {
            out[i] = len2;
        } else if len2 == 1 {
            out[i] = len1;
        } else {
            return Err(from_kind(ErrorKind::IncompatibleShape));
        }
    }
    Ok(out)
}

pub trait CompareDimensions {
    type Smallest;
    type Largest;
    /// Determines the shape after broadcasting the dimensions together.
    ///
    /// If the dimensions are not compatible, returns `Err`.
    ///
    /// Uses the [NumPy broadcasting rules]
    /// (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
    fn cobroadcast(&self) -> Result<Self::Largest, ShapeError>;
}

macro_rules! impl_comparedimensions_pair_identity {
    ($dim:ty) => {
        impl CompareDimensions for ($dim, $dim) {
            type Smallest = $dim;
            type Largest = $dim;
            fn cobroadcast(&self) -> Result<$dim, ShapeError> {
                cobroadcast_pair(&self.0, &self.1)
            }
        }
    }
}

impl_comparedimensions_pair_identity!(Ix0);
impl_comparedimensions_pair_identity!(Ix1);
impl_comparedimensions_pair_identity!(Ix2);
impl_comparedimensions_pair_identity!(Ix3);
impl_comparedimensions_pair_identity!(Ix4);
impl_comparedimensions_pair_identity!(Ix5);
impl_comparedimensions_pair_identity!(Ix6);
impl_comparedimensions_pair_identity!(IxDyn);

macro_rules! impl_comparedimensions_pair {
    ($smaller:ty, $larger:ty) => {
        impl CompareDimensions for ($smaller, $larger) {
            type Smallest = $smaller;
            type Largest = $larger;
            fn cobroadcast(&self) -> Result<$larger, ShapeError> {
                cobroadcast_pair(&self.0, &self.1)
            }
        }

        impl CompareDimensions for ($larger, $smaller) {
            type Smallest = $smaller;
            type Largest = $larger;
            fn cobroadcast(&self) -> Result<$larger, ShapeError> {
                cobroadcast_pair(&self.0, &self.1)
            }
        }
    }
}

impl_comparedimensions_pair!(Ix0, Ix1);
impl_comparedimensions_pair!(Ix0, Ix2);
impl_comparedimensions_pair!(Ix0, Ix3);
impl_comparedimensions_pair!(Ix0, Ix4);
impl_comparedimensions_pair!(Ix0, Ix5);
impl_comparedimensions_pair!(Ix0, Ix6);
impl_comparedimensions_pair!(Ix1, Ix2);
impl_comparedimensions_pair!(Ix1, Ix3);
impl_comparedimensions_pair!(Ix1, Ix4);
impl_comparedimensions_pair!(Ix1, Ix5);
impl_comparedimensions_pair!(Ix1, Ix6);
impl_comparedimensions_pair!(Ix2, Ix3);
impl_comparedimensions_pair!(Ix2, Ix4);
impl_comparedimensions_pair!(Ix2, Ix5);
impl_comparedimensions_pair!(Ix2, Ix6);
impl_comparedimensions_pair!(Ix3, Ix4);
impl_comparedimensions_pair!(Ix3, Ix5);
impl_comparedimensions_pair!(Ix3, Ix6);
impl_comparedimensions_pair!(Ix4, Ix5);
impl_comparedimensions_pair!(Ix4, Ix6);
impl_comparedimensions_pair!(Ix5, Ix6);

macro_rules! impl_comparedimensions_pair_dyn {
    ($other:ty) => {
        impl CompareDimensions for ($other, IxDyn) {
            type Smallest = IxDyn;
            type Largest = IxDyn;
            fn cobroadcast(&self) -> Result<IxDyn, ShapeError> {
                cobroadcast_pair(&self.0, &self.1)
            }
        }

        impl CompareDimensions for (IxDyn, $other) {
            type Smallest = IxDyn;
            type Largest = IxDyn;
            fn cobroadcast(&self) -> Result<IxDyn, ShapeError> {
                cobroadcast_pair(&self.0, &self.1)
            }
        }
    }
}

impl_comparedimensions_pair_dyn!(Ix0);
impl_comparedimensions_pair_dyn!(Ix1);
impl_comparedimensions_pair_dyn!(Ix2);
impl_comparedimensions_pair_dyn!(Ix3);
impl_comparedimensions_pair_dyn!(Ix4);
impl_comparedimensions_pair_dyn!(Ix5);
impl_comparedimensions_pair_dyn!(Ix6);
