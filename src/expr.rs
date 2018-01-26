// Copyright 2018 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides types to create and evaluate expression trees of
//! operations on arrays.
//!
//! ## Examples
//!
//! This is an example illustrating how to create and evaluate a simple
//! expression:
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//!
//! use ndarray::prelude::*;
//! use ndarray::expr::{Expression, ExpressionExt};
//!
//! # fn main() {
//! let a: Array1<i32> = array![1, 2, 3];
//! let b: Array1<i32> = array![4, -5, 6];
//! let c: Array1<i32> = array![-3, 7, -5];
//! assert_eq!(
//!     (-a.as_expr() * b.as_expr() + c.expr_map(|x| x.pow(2)) * a.as_expr()).eval(),
//!     array![5, 108, 57],
//! );
//! # }
//! ```
//!
//! This example illustrates cobroadcasting of two 2D arrays:
//!
//! ```
//! # #[macro_use(array)]
//! # extern crate ndarray;
//! # use ndarray::expr::{Expression, ExpressionExt};
//! # fn main() {
//! let a = array![
//!     [1],
//!     [2],
//!     [3]
//! ];
//! let b = array![
//!     [4, -5, 6]
//! ];
//! assert_eq!(
//!     (a.as_expr() + b.as_expr()).eval(),
//!     array![
//!         [5, -4, 7],
//!         [6, -3, 8],
//!         [7, -2, 9]
//!     ],
//! );
//! # }
//! ```
//!
//! This example illustrates cobroadcasting of a 3D array and a 2D array:
//!
//! ```
//! # #[macro_use(array)]
//! # extern crate ndarray;
//! # use ndarray::expr::{Expression, ExpressionExt};
//! # fn main() {
//! let a = array![[[1], [2], [3]], [[-7], [3], [-9]]].into_dyn();
//! let b = array![[4, -5, 6]].into_dyn();
//! assert_eq!(
//!     (a.as_expr() - b.as_expr()).eval(),
//!     array![
//!         [[-3, 6, -5], [-2, 7, -4], [-1, 8, -3]],
//!         [[-11, -2, -13], [-1, 8, -3], [-13, -4, -15]]
//!     ].into_dyn(),
//! );
//! # }
//! ```

use NdProducer;
use imp_prelude::*;
use layout::{Layout, LayoutPriv, CORDER, FORDER};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};
use zip::{Offset, Zippable};

/// An expression of arrays.
pub trait Expression: Zippable {
    /// Type of elements of output array.
    type OutElem: Copy;

    /// Returns the number of dimensions (axes) of the result.
    fn ndim(&self) -> usize;

    /// Returns a clone of the raw dimension.
    fn raw_dim(&self) -> Self::Dim;

    /// Returns the shape of the result.
    fn shape(&self) -> &[usize];

    /// Returns the total number of elements in the result.
    fn len(&self) -> usize;

    /// Returns the layout of the input arrays.
    fn layout(&self) -> Layout;

    /// Broadcast into a larger shape, if possible.
    fn broadcast_move(self, shape: Self::Dim) -> Option<Self>;

    /// Applies the expression to individual elements of the arrays.
    fn eval_item(&self, item: <Self as Zippable>::Item) -> Self::OutElem;

    /// Applies the expression to the arrays, returning the result.
    ///
    /// This method does not allocate any intermediate arrays; it allocates
    /// only the single output array.
    fn eval(&self) -> Array<Self::OutElem, Self::Dim> {
        let mut out = unsafe { Array::uninitialized(self.raw_dim()) };
        self.eval_assign(out.view_mut());
        out
    }

    /// Applies the expression to the arrays, assigning the result to `out`.
    fn eval_assign(&self, out: ArrayViewMut<Self::OutElem, Self::Dim>) {
        if self.layout().and(out.layout()).is(CORDER | FORDER) {
            self.eval_assign_contiguous(out)
        } else {
            self.eval_assign_strided(out)
        }
    }

    /// Creates an expression that calls `f` by value on each element.
    fn map_into<F, O>(self, f: F) -> UnaryOpExpr<F, Self, O>
    where
        F: Fn(Self::OutElem) -> O,
    {
        UnaryOpExpr::new(f, self)
    }
}

trait ExpressionPriv: Expression {
    fn eval_assign_contiguous(&self, out: ArrayViewMut<Self::OutElem, Self::Dim>) {
        assert_eq!(
            self.shape(),
            out.shape(),
            "Output array shape must match expression shape.",
        );
        debug_assert!(self.layout().and(out.layout()).is(CORDER | FORDER));
        let ptrs = (Zippable::as_ptr(&out), self.as_ptr());
        let inner_strides = (Zippable::contiguous_stride(&out), self.contiguous_stride());
        for i in 0..self.len() {
            unsafe {
                let (out_ptr_i, ptr_i) = ptrs.stride_offset(inner_strides, i);
                *Zippable::as_ref(&out, out_ptr_i) = self.eval_item(self.as_ref(ptr_i));
            }
        }
    }

    fn eval_assign_strided(&self, out: ArrayViewMut<Self::OutElem, Self::Dim>) {
        assert_eq!(
            self.shape(),
            out.shape(),
            "Output array shape must match expression shape.",
        );
        let n = self.ndim();
        debug_assert_ne!(n, 0, "Unreachable: ndim == 0 is contiguous");
        let unroll_axis = Axis(n - 1);
        let mut dim = self.raw_dim();
        let inner_len = dim[unroll_axis.index()];
        dim[unroll_axis.index()] = 1;
        let mut index_ = dim.first_index();
        let inner_strides = (
            Zippable::stride_of(&out, unroll_axis),
            self.stride_of(unroll_axis),
        );
        while let Some(index) = index_ {
            // Let's “unroll” the loop over the innermost axis
            unsafe {
                let ptr = (Zippable::uget_ptr(&out, &index), self.uget_ptr(&index));
                for i in 0..inner_len {
                    let (out_p, p) = ptr.stride_offset(inner_strides, i);
                    *Zippable::as_ref(&out, out_p) = self.eval_item(self.as_ref(p));
                }
            }
            index_ = dim.next_for(index);
        }
    }
}

impl<T> ExpressionPriv for T
where
    T: Expression,
{
}

/// Convenience extension methods for `ArrayBase`.
pub trait ExpressionExt<A, D>
where
    D: Dimension,
{
    /// Creates an expression view of `self`.
    fn as_expr(&self) -> ArrayViewExpr<A, D>;

    /// Creates an expression that calls `f` by value on each element.
    fn expr_map<'a, F, O>(&'a self, f: F) -> UnaryOpExpr<F, ArrayViewExpr<'a, A, D>, O>
    where
        F: Fn(&'a A) -> O,
        A: Copy;
}

impl<A, S, D> ExpressionExt<A, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn as_expr(&self) -> ArrayViewExpr<A, D> {
        ArrayViewExpr::new(self.view())
    }

    fn expr_map<'a, F, O>(&'a self, f: F) -> UnaryOpExpr<F, ArrayViewExpr<'a, A, D>, O>
    where
        F: Fn(&'a A) -> O,
        A: Copy,
    {
        self.as_expr().map_into(f)
    }
}

/// An expression wrapper for an `ArrayView`.
#[derive(Clone, Debug)]
pub struct ArrayViewExpr<'a, A: 'a, D: 'a>(ArrayView<'a, A, D>)
where
    D: Dimension;

impl<'a, A, D> ArrayViewExpr<'a, A, D>
where
    D: Dimension,
{
    /// Creates a new expression from the view.
    pub fn new(view: ArrayView<'a, A, D>) -> Self {
        ArrayViewExpr(view)
    }
}

impl<'a, A, D> Expression for ArrayViewExpr<'a, A, D>
where
    A: Copy,
    D: Dimension,
{
    type OutElem = &'a A;

    #[inline]
    fn ndim(&self) -> usize {
        self.0.ndim()
    }

    #[inline]
    fn raw_dim(&self) -> D {
        self.0.raw_dim()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn layout(&self) -> Layout {
        Layout::from(NdProducer::layout(&self.0))
    }

    fn broadcast_move(self, shape: D) -> Option<Self> {
        self.0.broadcast(shape.clone()).map(|bc| {
            // Copy strides from broadcasted view.
            let mut strides = D::zero_index_with_ndim(shape.ndim());
            strides
                .slice_mut()
                .iter_mut()
                .zip(bc.strides())
                .for_each(|(out_s, bc_s)| *out_s = *bc_s as usize);
            // Create a new `ArrayView` with the shape and strides. This is the
            // only way to keep the same lifetime since there is no
            // `broadcast_move` method for `ArrayView`.
            ArrayViewExpr(unsafe { ArrayView::from_shape_ptr(shape.strides(strides), bc.as_ptr()) })
        })
    }

    #[inline]
    fn eval_item(&self, item: &'a A) -> &'a A {
        item
    }
}

impl<'a, A, D> Zippable for ArrayViewExpr<'a, A, D>
where
    A: Copy,
    D: Dimension,
{
    type Item = &'a A;
    // TODO: why is this *mut and not *const?
    type Ptr = *mut A;
    type Dim = D;
    type Stride = isize;
    #[inline]
    fn as_ptr(&self) -> Self::Ptr {
        NdProducer::as_ptr(&self.0)
    }
    #[inline]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        NdProducer::as_ref(&self.0, ptr)
    }
    #[inline]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        NdProducer::uget_ptr(&self.0, i)
    }
    #[inline]
    fn stride_of(&self, axis: Axis) -> Self::Stride {
        NdProducer::stride_of(&self.0, axis)
    }
    #[inline]
    fn contiguous_stride(&self) -> Self::Stride {
        NdProducer::contiguous_stride(&self.0)
    }
    #[inline]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        let split = NdProducer::split_at(self.0, axis, index);
        (ArrayViewExpr(split.0), ArrayViewExpr(split.1))
    }
}

/// An expression with a single argument.
#[derive(Clone, Debug)]
pub struct UnaryOpExpr<F, E, O>
where
    F: Fn(E::OutElem) -> O,
    E: Expression,
{
    oper: F,
    inner: E,
}

impl<F, E, O> UnaryOpExpr<F, E, O>
where
    F: Fn(E::OutElem) -> O,
    E: Expression,
{
    /// Returns a new expression applying `oper` to `inner`.
    pub fn new(oper: F, inner: E) -> Self {
        UnaryOpExpr { oper, inner }
    }
}

impl<F, E, O> Expression for UnaryOpExpr<F, E, O>
where
    F: Fn(E::OutElem) -> O,
    E: Expression,
    O: Copy,
{
    type OutElem = O;

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
        let UnaryOpExpr { oper, inner, .. } = self;
        inner
            .broadcast_move(shape)
            .map(|new_inner| UnaryOpExpr::new(oper, new_inner))
    }

    #[inline]
    fn eval_item(&self, item: E::Item) -> O {
        (self.oper)(self.inner.eval_item(item))
    }
}

impl<F, E, O> Zippable for UnaryOpExpr<F, E, O>
where
    F: Fn(E::OutElem) -> O,
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
        //         oper: self.oper.clone(),
        //         inner: inner_split.0,
        //     },
        //     UnaryOpExpr {
        //         oper: self.oper,
        //         inner: inner_split.1,
        //     },
        // )
    }
}

/// Broadcast the shapes together, follwing the behavior of NumPy.
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

/// An expression with two arguments.
#[derive(Clone, Debug)]
pub struct BinaryOpExpr<F, E1, E2, O>
where
    F: Fn(E1::OutElem, E2::OutElem) -> O,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
{
    oper: F,
    left: E1,
    right: E2,
}

impl<F, E1, E2, O> BinaryOpExpr<F, E1, E2, O>
where
    F: Fn(E1::OutElem, E2::OutElem) -> O,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
{
    /// Returns a new expression applying `oper` to `left` and `right`.
    ///
    /// Returns `None` if the shapes of the arrays cannot be broadcast
    /// together. Note that the broadcasting is more general than
    /// `ArrayBase.broadcast()`; cobroadcasting is supported.
    pub fn new(oper: F, left: E1, right: E2) -> Option<Self> {
        broadcast::<E1::Dim>(left.shape(), right.shape()).map(|shape| {
            let left = left.broadcast_move(shape.clone()).unwrap();
            let right = right.broadcast_move(shape.clone()).unwrap();
            BinaryOpExpr { oper, left, right }
        })
    }
}

impl<F, E1, E2, O> Expression for BinaryOpExpr<F, E1, E2, O>
where
    F: Clone + Fn(E1::OutElem, E2::OutElem) -> O,
    E1: Expression,
    E2: Expression<Dim = E1::Dim>,
    O: Copy,
{
    type OutElem = O;

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
            oper, left, right, ..
        } = self;
        match (
            left.broadcast_move(shape.clone()),
            right.broadcast_move(shape),
        ) {
            (Some(new_left), Some(new_right)) => BinaryOpExpr::new(oper, new_left, new_right),
            _ => None,
        }
    }

    #[inline]
    fn eval_item(&self, (left_item, right_item): (E1::Item, E2::Item)) -> O {
        (self.oper)(
            self.left.eval_item(left_item),
            self.right.eval_item(right_item),
        )
    }
}

impl<F, E1, E2, O> Zippable for BinaryOpExpr<F, E1, E2, O>
where
    F: Clone + Fn(E1::OutElem, E2::OutElem) -> O,
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
        //         oper: self.oper.clone(),
        //         left: left_split.0,
        //         right: right_split.0,
        //     },
        //     BinaryOpExpr {
        //         oper: self.oper,
        //         left: left_split.1,
        //         right: right_split.1,
        //     },
        // )
    }
}

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident, ($($header:tt)*), ($($constraints:tt)*)) => {
        $($header)*
        where
            Self: Expression,
            <Self as Expression>::OutElem: $trait,
            $($constraints)*
        {
            type Output = UnaryOpExpr<
                fn(<Self as Expression>::OutElem)
                    -> <<Self as Expression>::OutElem as $trait>::Output,
                Self,
                <<Self as Expression>::OutElem as $trait>::Output,
            >;

            #[inline(always)]
            fn $method(self) -> Self::Output {
                UnaryOpExpr::new($trait::$method, self)
            }
        }
    }
}

macro_rules! impl_unary_op_all {
    ($trait:ident, $method:ident) => {
        impl_unary_op!(
            $trait, $method,
            (impl<'a, A, D> $trait for ArrayViewExpr<'a, A, D>),
            (D: Dimension)
        );
        impl_unary_op!(
            $trait, $method,
            (impl<F, E, O> $trait for UnaryOpExpr<F, E, O>),
            (
                F: Fn(E::OutElem) -> O,
                E: Expression,
            )
        );
        impl_unary_op!(
            $trait, $method,
            (impl<F, E1, E2, O> $trait for BinaryOpExpr<F, E1, E2, O>),
            (
                F: Fn(E1::OutElem, E2::OutElem) -> O,
                E1: Expression,
                E2: Expression<Dim = E1::Dim>,
            )
        );
    }
}

impl_unary_op_all!(Neg, neg);
impl_unary_op_all!(Not, not);

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, ($($header:tt)*), ($($constraints:tt)*)) => {
        $($header)*
        where
            Self: Expression,
            <Self as Expression>::OutElem: $trait<Rhs::OutElem>,
            Rhs: Expression<Dim = <Self as Zippable>::Dim>,
            $($constraints)*
        {
            type Output = BinaryOpExpr<
                fn(<Self as Expression>::OutElem, Rhs::OutElem)
                    -> <<Self as Expression>::OutElem as $trait<Rhs::OutElem>>::Output,
                Self,
                Rhs,
                <<Self as Expression>::OutElem as $trait<<Rhs as Expression>::OutElem>>::Output,
            >;

            #[inline(always)]
            fn $method(self, rhs: Rhs) -> Self::Output {
                // Extra type annotation is necessary to prevent compile error
                // due to incorrect inference.
                BinaryOpExpr::<fn(_, _) -> _, _, _, _>::new($trait::$method, self, rhs).unwrap()
            }
        }
    }
}

macro_rules! impl_binary_op_all {
    ($trait:ident, $method:ident) => {
        impl_binary_op!(
            $trait, $method,
            (impl<'a, A, D, Rhs> $trait<Rhs> for ArrayViewExpr<'a, A, D>),
            (D: Dimension)
        );
        impl_binary_op!(
            $trait, $method,
            (impl<F, E, O, Rhs> $trait<Rhs> for UnaryOpExpr<F, E, O>),
            (
                F: Fn(E::OutElem) -> O,
                E: Expression,
            )
        );
        impl_binary_op!(
            $trait, $method,
            (impl<F, E1, E2, O, Rhs> $trait<Rhs> for BinaryOpExpr<F, E1, E2, O>),
            (
                F: Fn(E1::OutElem, E2::OutElem) -> O,
                E1: Expression,
                E2: Expression<Dim = E1::Dim>,
            )
        );
    }
}

impl_binary_op_all!(Add, add);
impl_binary_op_all!(BitAnd, bitand);
impl_binary_op_all!(BitOr, bitor);
impl_binary_op_all!(BitXor, bitxor);
impl_binary_op_all!(Div, div);
impl_binary_op_all!(Mul, mul);
impl_binary_op_all!(Rem, rem);
impl_binary_op_all!(Sub, sub);
