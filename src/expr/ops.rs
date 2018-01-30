use expr::{ArrayViewExpr, Expression, ExpressionPriv, FnExpr};
use imp_prelude::*;
use layout::{Layout, LayoutPriv};
use std::marker::PhantomData;
use zip::Zippable;

fn broadcast<D: Dimension>(shape1: &[usize], shape2: &[usize]) -> Option<D> {
    // Zip the dims in reverse order, adding `&1`s to the shorter one until
    // they're the same length.
    let zipped = if shape1.len() < shape2.len() {
        shape1
            .iter()
            .rev()
            .chain(::std::iter::repeat(&1))
            .zip(shape2.iter().rev())
    } else {
        shape2
            .iter()
            .rev()
            .chain(::std::iter::repeat(&1))
            .zip(shape1.iter().rev())
    };
    let mut out = D::zero_index_with_ndim(::std::cmp::max(shape1.len(), shape2.len()));
    for ((&len1, &len2), len_out) in zipped.zip(out.slice_mut().iter_mut().rev()) {
        if len1 == len2 {
            *len_out = len1;
        } else if len1 == 1 {
            *len_out = len2;
        } else if len2 == 1 {
            *len_out = len1;
        } else {
            return None;
        }
    }
    Some(out)
}

pub trait UnaryOperator<Arg> {
    type Output;
    fn call(arg: Arg) -> Self::Output;
}

/// An expression with a single argument.
#[derive(Clone, Debug)]
pub struct UnaryOpExpr<F, E>
where
    F: UnaryOperator<E::OutElem>,
    E: Expression,
{
    op: F,
    inner: E,
}

impl<F, E> UnaryOpExpr<F, E>
where
    F: UnaryOperator<E::OutElem>,
    E: Expression,
{
    /// Returns a new expression applying `op` to `inner`.
    pub fn new(op: F, inner: E) -> Self {
        UnaryOpExpr { op, inner }
    }
}

impl<F, E> Expression for UnaryOpExpr<F, E>
where
    F: UnaryOperator<E::OutElem>,
    E: Expression,
    F::Output: Copy,
{
    type OutElem = F::Output;

    #[inline]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[inline]
    fn raw_dim(&self) -> Self::Dim {
        self.inner.raw_dim()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn layout(&self) -> Layout {
        self.inner.layout()
    }

    fn broadcast_move(self, shape: Self::Dim) -> Option<Self> {
        let UnaryOpExpr { op, inner, .. } = self;
        inner
            .broadcast_move(shape)
            .map(|new_inner| UnaryOpExpr::new(op, new_inner))
    }

    #[inline]
    fn eval_item(&self, item: E::Item) -> F::Output {
        F::call(self.inner.eval_item(item))
    }
}

impl<F, E> Zippable for UnaryOpExpr<F, E>
where
    F: UnaryOperator<E::OutElem>,
    E: Expression,
{
    type Item = E::Item;
    type Ptr = E::Ptr;
    type Dim = E::Dim;
    type Stride = E::Stride;
    #[inline]
    fn stride_of(&self, axis: Axis) -> Self::Stride {
        self.inner.stride_of(axis)
    }
    #[inline]
    fn contiguous_stride(&self) -> Self::Stride {
        self.inner.contiguous_stride()
    }
    #[inline]
    fn as_ptr(&self) -> Self::Ptr {
        self.inner.as_ptr()
    }
    #[inline]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        self.inner.as_ref(ptr)
    }
    #[inline]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.inner.uget_ptr(i)
    }
    #[inline]
    fn split_at(self, _axis: Axis, _index: usize) -> (Self, Self) {
        unimplemented!()
        // let inner_split = self.inner.split_at(axis, index);
        // (
        //     UnaryOpExpr {
        //         op: self.op.clone(),
        //         inner: inner_split.0,
        //     },
        //     UnaryOpExpr {
        //         op: self.op,
        //         inner: inner_split.1,
        //     },
        // )
    }
}

pub trait BinaryOperator<Lhs, Rhs> {
    type Output;
    fn call(lhs: Lhs, rhs: Rhs) -> Self::Output;
}

/// An expression with two arguments.
#[derive(Clone, Debug)]
pub struct BinaryOpExpr<F, E1, E2>
where
    F: BinaryOperator<E1::OutElem, E2::OutElem>,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
{
    op: F,
    left: E1,
    right: E2,
}

impl<F, E1, E2> BinaryOpExpr<F, E1, E2>
where
    F: BinaryOperator<E1::OutElem, E2::OutElem>,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
{
    /// Returns a new expression applying `op` to `left` and `right`.
    ///
    /// Returns `None` if the shapes of the arrays cannot be broadcast
    /// together. Note that the broadcasting is more general than
    /// `ArrayBase.broadcast()`; cobroadcasting is supported.
    pub fn new(op: F, left: E1, right: E2) -> Option<Self> {
        broadcast::<E1::Dim>(left.shape(), right.shape()).map(|shape| {
            let left = left.broadcast_move(shape.clone()).unwrap();
            let right = right.broadcast_move(shape.clone()).unwrap();
            BinaryOpExpr { op, left, right }
        })
    }
}

impl<F, E1, E2> Expression for BinaryOpExpr<F, E1, E2>
where
    F: BinaryOperator<E1::OutElem, E2::OutElem>,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
    F::Output: Copy,
{
    type OutElem = F::Output;

    #[inline]
    fn ndim(&self) -> usize {
        self.left.ndim()
    }

    #[inline]
    fn raw_dim(&self) -> Self::Dim {
        self.left.raw_dim()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        self.left.shape()
    }

    #[inline]
    fn len(&self) -> usize {
        self.left.len()
    }

    #[inline]
    fn layout(&self) -> Layout {
        self.left.layout().and(self.right.layout())
    }

    fn broadcast_move(self, shape: Self::Dim) -> Option<Self> {
        let BinaryOpExpr {
            op, left, right, ..
        } = self;
        match (
            left.broadcast_move(shape.clone()),
            right.broadcast_move(shape),
        ) {
            (Some(new_left), Some(new_right)) => BinaryOpExpr::new(op, new_left, new_right),
            _ => None,
        }
    }

    #[inline]
    fn eval_item(&self, (left_item, right_item): (E1::Item, E2::Item)) -> F::Output {
        F::call(
            self.left.eval_item(left_item),
            self.right.eval_item(right_item),
        )
    }
}

impl<F, E1, E2> Zippable for BinaryOpExpr<F, E1, E2>
where
    F: BinaryOperator<E1::OutElem, E2::OutElem>,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
{
    type Item = (E1::Item, E2::Item);
    type Ptr = (E1::Ptr, E2::Ptr);
    type Dim = E1::Dim;
    type Stride = (E1::Stride, E2::Stride);
    #[inline]
    fn stride_of(&self, axis: Axis) -> Self::Stride {
        (self.left.stride_of(axis), self.right.stride_of(axis))
    }
    #[inline]
    fn contiguous_stride(&self) -> Self::Stride {
        (
            self.left.contiguous_stride(),
            self.right.contiguous_stride(),
        )
    }
    #[inline]
    fn as_ptr(&self) -> Self::Ptr {
        (self.left.as_ptr(), self.right.as_ptr())
    }
    #[inline]
    unsafe fn as_ref(&self, (left_ptr, right_ptr): (E1::Ptr, E2::Ptr)) -> Self::Item {
        (self.left.as_ref(left_ptr), self.right.as_ref(right_ptr))
    }
    #[inline]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        (self.left.uget_ptr(i), self.right.uget_ptr(i))
    }
    #[inline]
    fn split_at(self, _axis: Axis, _index: usize) -> (Self, Self) {
        unimplemented!()
        // let left_split = self.left.split_at(axis, index);
        // let right_split = self.right.split_at(axis, index);
        // (
        //     BinaryOpExpr {
        //         op: self.op.clone(),
        //         left: left_split.0,
        //         right: right_split.0,
        //     },
        //     BinaryOpExpr {
        //         op: self.op,
        //         left: left_split.1,
        //         right: right_split.1,
        //     },
        // )
    }
}

/// Define structs that call unary operators (implement `UnaryOperator`).
macro_rules! define_unary_op {
    ($trait:ident, $method:ident) => {
        #[derive(Clone, Debug)]
        pub struct $trait<Arg>
        where
            Arg: ::std::ops::$trait,
        {
            arg: PhantomData<Arg>,
        }

        impl<Arg> $trait<Arg>
        where
            Arg: ::std::ops::$trait,
        {
            pub fn new() -> Self {
                $trait {
                    arg: PhantomData,
                }
            }
        }

        impl<Arg> UnaryOperator<Arg> for $trait<Arg>
        where
            Arg: ::std::ops::$trait,
        {
            type Output = <Arg as ::std::ops::$trait>::Output;

            fn call(arg: Arg) -> Self::Output {
                ::std::ops::$trait::$method(arg)
            }
        }
    }
}

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident, ($($header:tt)*), ($($constraints:tt)*)) => {
        $($header)*
        where
            <Self as Expression>::OutElem: ::std::ops::$trait,
            $($constraints)*
        {
            type Output = UnaryOpExpr<$trait<<Self as Expression>::OutElem>, Self>;

            #[inline(always)]
            fn $method(self) -> Self::Output {
                UnaryOpExpr::new($trait::new(), self)
            }
        }
    }
}

macro_rules! impl_unary_op_for_fn_expr {
    ($trait:ident, $method:ident, ($($generics:ident),*)) => {
        impl_unary_op!(
            $trait, $method,
            (impl<Dim, O, F, $($generics),*> ::std::ops::$trait for FnExpr<F, ($($generics),*,)>),
            (
                Dim: Dimension,
                O: Copy,
                F: Fn($($generics::OutElem),*) -> O,
                $($generics: Expression<Dim = Dim>),*
            )
        );
    }
}

/// For an operator, define the corresponding `UnaryOperator` struct and
/// implement the operator for all the `Expression` types.
macro_rules! define_and_impl_unary_op {
    ($trait:ident, $method:ident) => {
        define_unary_op!($trait, $method);
        impl_unary_op!(
            $trait, $method,
            (impl<'a, A, D> ::std::ops::$trait for ArrayViewExpr<'a, A, D>),
            (
                A: Copy,
                D: Dimension,
            )
        );
        impl_unary_op!(
            $trait, $method,
            (impl<F, E> ::std::ops::$trait for UnaryOpExpr<F, E>),
            (
                F: UnaryOperator<E::OutElem>,
                E: Expression,
                F::Output: Copy,
            )
        );
        impl_unary_op!(
            $trait, $method,
            (impl<F, E1, E2> ::std::ops::$trait for BinaryOpExpr<F, E1, E2>),
            (
                F: BinaryOperator<E1::OutElem, E2::OutElem>,
                E1: Expression,
                E2: Expression<Dim = E1::Dim>,
                F::Output: Copy,
            )
        );
        impl_unary_op_for_fn_expr!($trait, $method, (E1));
        impl_unary_op_for_fn_expr!($trait, $method, (E1, E2));
        impl_unary_op_for_fn_expr!($trait, $method, (E1, E2, E3));
        impl_unary_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4));
        impl_unary_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4, E5));
        impl_unary_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4, E5, E6));
    }
}

define_and_impl_unary_op!(Neg, neg);
define_and_impl_unary_op!(Not, not);

/// Define structs that call binary operators (implement `BinaryOperator`).
macro_rules! define_binary_op {
    ($trait:ident, $method:ident) => {
        #[derive(Clone, Debug)]
        pub struct $trait<Lhs, Rhs>
        where
            Lhs: ::std::ops::$trait<Rhs>,
        {
            lhs: PhantomData<Lhs>,
            rhs: PhantomData<Rhs>,
        }

        impl<Lhs, Rhs> $trait<Lhs, Rhs>
        where
            Lhs: ::std::ops::$trait<Rhs>,
        {
            pub fn new() -> Self {
                $trait {
                    lhs: PhantomData,
                    rhs: PhantomData,
                }
            }
        }

        impl<Lhs, Rhs> BinaryOperator<Lhs, Rhs> for $trait<Lhs, Rhs>
        where
            Lhs: ::std::ops::$trait<Rhs>,
        {
            type Output = <Lhs as ::std::ops::$trait<Rhs>>::Output;

            fn call(lhs: Lhs, rhs: Rhs) -> Self::Output {
                ::std::ops::$trait::$method(lhs, rhs)
            }
        }
    }
}

/// Implement a binary operator for an `Expression` type.
macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, ($($header:tt)*), ($($constraints:tt)*)) => {
        $($header)*
        where
            <Self as Expression>::OutElem: ::std::ops::$trait<Rhs::OutElem>,
            Rhs: Expression<Dim = <Self as Zippable>::Dim>,
            $($constraints)*
        {
            type Output = BinaryOpExpr<
                $trait<<Self as Expression>::OutElem, Rhs::OutElem>,
                Self,
                Rhs,
            >;

            #[inline(always)]
            fn $method(self, rhs: Rhs) -> Self::Output {
                BinaryOpExpr::new(
                    $trait::new(), self, rhs
                ).unwrap()
            }
        }
    }
}

macro_rules! impl_binary_op_for_fn_expr {
    ($trait:ident, $method:ident, ($($generics:ident),*)) => {
        impl_binary_op!(
            $trait, $method,
            (impl<Rhs, Dim, O, F, $($generics),*> ::std::ops::$trait<Rhs>
             for FnExpr<F, ($($generics),*,)>),
            (
                Dim: Dimension,
                O: Copy,
                F: Fn($($generics::OutElem),*) -> O,
                $($generics: Expression<Dim = Dim>),*
            )
        );
    }
}

/// For an operator, define the corresponding `BinaryOperator` struct and
/// implement the operator for all the `Expression` types.
macro_rules! define_and_impl_binary_op {
    ($trait:ident, $method:ident) => {
        define_binary_op!($trait, $method);
        impl_binary_op!(
            $trait, $method,
            (impl<'a, A, D, Rhs> ::std::ops::$trait<Rhs> for ArrayViewExpr<'a, A, D>),
            (
                A: Copy,
                D: Dimension
            )
        );
        impl_binary_op!(
            $trait, $method,
            (impl<F, E, Rhs> ::std::ops::$trait<Rhs> for UnaryOpExpr<F, E>),
            (
                F: UnaryOperator<E::OutElem>,
                E: Expression,
                F::Output: Copy,
            )
        );
        impl_binary_op!(
            $trait, $method,
            (impl<F, E1, E2, Rhs> ::std::ops::$trait<Rhs> for BinaryOpExpr<F, E1, E2>),
            (
                F: BinaryOperator<E1::OutElem, E2::OutElem>,
                E1: Expression,
                E2: Expression<Dim = E1::Dim>,
                F::Output: Copy,
            )
        );
        impl_binary_op_for_fn_expr!($trait, $method, (E1));
        impl_binary_op_for_fn_expr!($trait, $method, (E1, E2));
        impl_binary_op_for_fn_expr!($trait, $method, (E1, E2, E3));
        impl_binary_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4));
        impl_binary_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4, E5));
        impl_binary_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4, E5, E6));
    }
}

define_and_impl_binary_op!(Add, add);
define_and_impl_binary_op!(BitAnd, bitand);
define_and_impl_binary_op!(BitOr, bitor);
define_and_impl_binary_op!(BitXor, bitxor);
define_and_impl_binary_op!(Div, div);
define_and_impl_binary_op!(Mul, mul);
define_and_impl_binary_op!(Rem, rem);
define_and_impl_binary_op!(Sub, sub);

macro_rules! impl_inplace_op {
    ($trait:ident, $method:ident, ($($generics:tt)*), $rhs:ty, $outelem:ty, ($($constraints:tt)*)) => {
        impl<$($generics)*, Ao, S, D> ::std::ops::$trait<$rhs> for ArrayBase<S, D>
        where
            Ao: ::std::ops::$trait<$outelem>,
            S: DataMut<Elem = Ao>,
            D: Dimension,
            $($constraints)*
        {
            fn $method(&mut self, rhs: $rhs) {
                rhs.eval_apply(
                    |out, res| ::std::ops::$trait::$method(out, res),
                    self.view_mut(),
                )
            }
        }
    }
}

macro_rules! impl_inplace_op_for_fn_expr {
    ($trait:ident, $method:ident, ($($generics:ident),*)) => {
        impl_inplace_op!(
            $trait, $method,
            (O, F, $($generics),*),
            FnExpr<F, ($($generics),*,)>,
            O,
            (
                O: Copy,
                F: Fn($($generics::OutElem),*) -> O,
                $($generics: Expression<Dim = D>),*
            )
        );
    }
}

macro_rules! impl_inplace_op_all {
    ($trait:ident, $method:ident) => {
        impl_inplace_op!(
            $trait, $method,
            ('a, Ai),
            ArrayViewExpr<'a, Ai, D>,
            &'a Ai,
            (Ai: Copy)
        );
        impl_inplace_op!(
            $trait, $method,
            (F, E),
            UnaryOpExpr<F, E>,
            F::Output,
            (
                F: UnaryOperator<E::OutElem>,
                E: Expression<Dim = D>,
                F::Output: Copy
            )
        );
        impl_inplace_op!(
            $trait, $method,
            (F, E1, E2),
            BinaryOpExpr<F, E1, E2>,
            F::Output,
            (
                F: BinaryOperator<E1::OutElem, E2::OutElem>,
                E1: Expression<Dim = D>,
                E2: Expression<Dim = D>,
                F::Output: Copy,
            )
        );
        impl_inplace_op_for_fn_expr!($trait, $method, (E1));
        impl_inplace_op_for_fn_expr!($trait, $method, (E1, E2));
        impl_inplace_op_for_fn_expr!($trait, $method, (E1, E2, E3));
        impl_inplace_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4));
        impl_inplace_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4, E5));
        impl_inplace_op_for_fn_expr!($trait, $method, (E1, E2, E3, E4, E5, E6));
    }
}

impl_inplace_op_all!(AddAssign, add_assign);
impl_inplace_op_all!(BitAndAssign, bitand_assign);
impl_inplace_op_all!(BitOrAssign, bitor_assign);
impl_inplace_op_all!(BitXorAssign, bitxor_assign);
impl_inplace_op_all!(DivAssign, div_assign);
impl_inplace_op_all!(MulAssign, mul_assign);
impl_inplace_op_all!(RemAssign, rem_assign);
impl_inplace_op_all!(ShlAssign, shl_assign);
impl_inplace_op_all!(ShrAssign, shr_assign);
impl_inplace_op_all!(SubAssign, sub_assign);
