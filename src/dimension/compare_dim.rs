use {Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

pub trait CompareDimensions {
    type Smallest;
    type Largest;
}

macro_rules! impl_comparedimensions_pair_identity {
    ($dim:ty) => {
        impl CompareDimensions for ($dim, $dim) {
            type Smallest = $dim;
            type Largest = $dim;
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
        }

        impl CompareDimensions for ($larger, $smaller) {
            type Smallest = $smaller;
            type Largest = $larger;
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
        }

        impl CompareDimensions for (IxDyn, $other) {
            type Smallest = IxDyn;
            type Largest = IxDyn;
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
