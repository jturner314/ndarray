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
use zip::{Offset, Zippable};

mod ops;

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
    fn map_into<F, O>(self, f: F) -> UnaryFnExpr<F, Self>
    where
        F: Fn(Self::OutElem) -> O,
    {
        UnaryFnExpr::new(f, self)
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
    fn expr_map<'a, F, O>(&'a self, f: F) -> UnaryFnExpr<F, ArrayViewExpr<'a, A, D>>
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

    fn expr_map<'a, F, O>(&'a self, f: F) -> UnaryFnExpr<F, ArrayViewExpr<'a, A, D>>
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

/// An expression that applies a function to an inner expression.
pub type UnaryFnExpr<F, E> = FnExpr<F, (E,)>;

/// An expression that applies a function to two inner expressions.
pub type BinaryFnExpr<F, E1, E2> = FnExpr<F, (E1, E2)>;

/// Broadcast the shapes together, follwing the behavior of NumPy.
fn multi_broadcast<D: Dimension>(shapes: &[&[usize]]) -> Option<D> {
    let ndim = match shapes.iter().map(|shape| shape.len()).max() {
        Some(max) => max,
        None => return None,
    };
    let mut out = D::zero_index_with_ndim(ndim);
    // axis = -1, -2, ...
    for axis in (-(ndim as isize)..0).rev() {
        let out_len = {
            // borrowck
            let out_axis = (out.ndim() as isize + axis) as usize;
            &mut out[out_axis]
        };
        *out_len = 1;
        for &shape in shapes {
            let shape_len = shape.get((shape.len() as isize + axis) as usize);
            match (*out_len, shape_len) {
                (_, None) => (),
                (1, Some(&len)) => *out_len = len,
                (eq, Some(&len)) if eq == len => (),
                (_, Some(_)) => return None,
            }
        }
    }
    Some(out)
}

/// An expression that applies a function to multiple argument expressions.
#[derive(Clone, Debug)]
pub struct FnExpr<F, Es> {
    f: F,
    inner: Es,
}

macro_rules! impl_fn_expr_new {
    (($generic:ident,), ($var:ident,)) => {
        impl<O, F, $generic> FnExpr<F, ($generic,)>
        where
            F: Fn($generic::OutElem) -> O,
            $generic: Expression,
        {
            /// Returns a new expression applying `f` to the argument.
            pub fn new(f: F, $var: $generic) -> Self {
                FnExpr {
                    f,
                    inner: ($var,),
                }
            }
        }
    };
    (($($generics:ident),*), ($($vars:ident),*)) => {
        impl<Dim, O, F, $($generics),*> FnExpr<F, ($($generics),*)>
        where
            Dim: Dimension,
            F: Fn($($generics::OutElem),*) -> O,
            $($generics: Expression<Dim = Dim>),*
        {
            /// Returns a new expression applying `f` to the arguments.
            ///
            /// Returns `None` if the shapes of the expressions cannot be
            /// broadcast together. Note that the broadcasting is more general
            /// than `ArrayBase.broadcast()`; cobroadcasting is supported.
            pub fn new(f: F, $($vars: $generics),*) -> Option<Self> {
                multi_broadcast::<Dim>(&[$($vars.shape()),*]).map(|shape| {
                    FnExpr {
                        f,
                        inner: (
                            $($vars.broadcast_move(shape.clone()).unwrap()),*
                        ),
                    }
                })
            }
        }
    };
}

macro_rules! impl_expression_for_fn_expr {
    (($gen_head:ident, $($gen_tail:ident),*), ($var_head:ident, $($var_tail:ident),*)) => {
        #[allow(non_snake_case)]
        impl<Dim, O, F, $gen_head, $($gen_tail),*> Expression
            for FnExpr<F, ($gen_head, $($gen_tail),*)>
        where
            Dim: Dimension,
            O: Copy,
            F: Fn($gen_head::OutElem, $($gen_tail::OutElem),*) -> O,
            $gen_head: Expression<Dim = Dim>,
            $($gen_tail: Expression<Dim = Dim>),*
        {
            type OutElem = O;

            #[inline]
            fn ndim(&self) -> usize {
                self.inner.0.ndim()
            }

            #[inline]
            fn raw_dim(&self) -> Self::Dim {
                self.inner.0.raw_dim()
            }

            #[inline]
            fn shape(&self) -> &[usize] {
                self.inner.0.shape()
            }

            #[inline]
            fn len(&self) -> usize {
                self.inner.0.len()
            }

            #[inline]
            fn layout(&self) -> Layout {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                $var_head.layout()$(.and($var_tail.layout()))*
            }

            fn broadcast_move(self, shape: Self::Dim) -> Option<Self> {
                let FnExpr {
                    f,
                    inner: ($var_head, $($var_tail),*),
                } = self;
                match (
                    $var_head.broadcast_move(shape.clone()),
                    $($var_tail.broadcast_move(shape.clone())),*
                ) {
                    (Some($var_head), $(Some($var_tail)),*) => {
                        Some(Self {
                            f,
                            inner: ($var_head, $($var_tail),*),
                        })
                    }
                    _ => None,
                }
            }

            #[inline]
            fn eval_item(&self, item: Self::Item) -> O {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                let ($gen_head, $($gen_tail),*) = item;
                (self.f)(
                    $var_head.eval_item($gen_head),
                    $($var_tail.eval_item($gen_tail)),*
                )
            }
        }
    }
}

macro_rules! impl_zippable_for_fn_expr {
    (($gen_head:ident, $($gen_tail:ident),*), ($var_head:ident, $($var_tail:ident),*)) => {
        #[allow(non_snake_case)]
        impl<Dim, O, F, $gen_head, $($gen_tail),*> Zippable
            for FnExpr<F, ($gen_head, $($gen_tail),*)>
        where
            Dim: Dimension,
            F: Fn($gen_head::OutElem, $($gen_tail::OutElem),*) -> O,
            $gen_head: Expression<Dim = Dim>,
            $($gen_tail: Expression<Dim = Dim>),*
        {
            type Item = ($gen_head::Item, $($gen_tail::Item),*);
            type Ptr = ($gen_head::Ptr, $($gen_tail::Ptr),*);
            type Dim = Dim;
            type Stride = ($gen_head::Stride, $($gen_tail::Stride),*);
            #[inline]
            fn stride_of(&self, axis: Axis) -> Self::Stride {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                ($var_head.stride_of(axis), $($var_tail.stride_of(axis)),*)
            }
            #[inline]
            fn contiguous_stride(&self) -> Self::Stride {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                ($var_head.contiguous_stride(), $($var_tail.contiguous_stride()),*)
            }
            #[inline]
            fn as_ptr(&self) -> Self::Ptr {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                ($var_head.as_ptr(), $($var_tail.as_ptr()),*)
            }
            #[inline]
            unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                let ($gen_head, $($gen_tail),*) = ptr;
                ($var_head.as_ref($gen_head), $($var_tail.as_ref($gen_tail)),*)
            }
            #[inline]
            unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
                let (ref $var_head, $(ref $var_tail),*) = self.inner;
                ($var_head.uget_ptr(i), $($var_tail.uget_ptr(i)),*)
            }
            #[inline]
            fn split_at(self, _axis: Axis, _index: usize) -> (Self, Self) {
                unimplemented!()
            }
        }
    }
}

macro_rules! impl_fn_expr {
    (($gen_head:ident, $($gen_tail:ident),*), ($var_head:ident, $($var_tail:ident),*)) => {
        impl_fn_expr_new!(($gen_head, $($gen_tail),*), ($var_head, $($var_tail),*));
        impl_expression_for_fn_expr!(($gen_head, $($gen_tail),*), ($var_head, $($var_tail),*));
        impl_zippable_for_fn_expr!(($gen_head, $($gen_tail),*), ($var_head, $($var_tail),*));
    }
}

impl_fn_expr!((E1,), (expr1,));
impl_fn_expr!((E1, E2), (expr1, expr2));
impl_fn_expr!((E1, E2, E3), (expr1, expr2, expr3));
impl_fn_expr!((E1, E2, E3, E4), (expr1, expr2, expr3, expr4));
impl_fn_expr!((E1, E2, E3, E4, E5), (expr1, expr2, expr3, expr4, expr5));
impl_fn_expr!(
    (E1, E2, E3, E4, E5, E6),
    (expr1, expr2, expr3, expr4, expr5, expr6)
);
