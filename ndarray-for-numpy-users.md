# `ndarray` for NumPy users

`ndarray`'s array type
([`ArrayBase`](https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html)), is
very similar to NumPy's array type (`numpy.ndarray`):

* Arrays have a single element type.
* Arrays can have arbitrarily many dimensions.
* Arrays can have arbitrary strides.
* Indexing starts at zero, not one.
* The default memory layout is row-major, and the default iterators follow
  row-major order (also called "logical order" in the documentation).
* Arithmetic operators work elementwise. (For example, `a * b` performs
  elementwise multiplication, not matrix multiplication.)
* Owned arrays are contiguous in memory.
* Many operations, such as slicing, are very cheap because they can return
  a view of an array instead of copying the data.

NumPy has many features that `ndarray` doesn't have yet, such as:

* [index arrays](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#index-arrays)
* [mask index arrays](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#boolean-or-mask-index-arrays)
* co-broadcasting (`ndarray` only supports broadcasting the right-hand array in a binary operation.)

## Some key differences

<table>
<tr>
<th>

NumPy

</th>
<th>

`ndarray`

</th>
</tr>

<tr>
<td>

In NumPy, there is no distinction between owned arrays, views, and mutable
views. There can be multiple arrays (instances of `numpy.ndarray`) that
mutably reference the same data.

</td>
<td>

In `ndarray`, all arrays are instances of
[`ArrayBase`](https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html), but
`ArrayBase` is generic over the ownership of the data.
[`Array`](https://docs.rs/ndarray/0.11/ndarray/type.Array.html) owns its data;
[`ArrayView`](https://docs.rs/ndarray/0.11/ndarray/type.ArrayView.html) is a
view;
[`ArrayViewMut`](https://docs.rs/ndarray/0.11/ndarray/type.ArrayViewMut.html)
is a mutable view; and
[`ArcArray`](https://docs.rs/ndarray/0.11/ndarray/type.ArcArray.html) has a
reference-counted pointer to its data (with copy-on-write mutation). Arrays and
views follow Rust's aliasing rules.

</td>
</tr>

<tr>
<td>

In NumPy, all arrays are dynamic-dimensional.

</td>
<td>

In `ndarray`, you can create fixed-dimension arrays, such as
[`Array2`](https://docs.rs/ndarray/0.11/ndarray/type.Array2.html). This takes
advantage of the type system to help you write correct code and also avoids
small heap allocations for the shape and strides.

</td>
</tr>

<tr>
<td>

When slicing in NumPy, the indices are `start`, `start + step`, `start +
2*step`, … until reaching `end` (exclusive).

</td>
<td>

When slicing in `ndarray`, the axis is first sliced with `start..end`. Then if
`step` is positive, the first index is the front of the slice; if `step` is
negative, the first index is the back of the slice. This means that the
behavior is the same as NumPy except when `step < -1`. See the docs for the
[`s![]` macro][s!] for more details.

</td>
</tr>
</table>

## Rough `ndarray`–NumPy equivalents

These tables provide some rough equivalents of NumPy operations in `ndarray`.
There are a variety of other methods that aren't included in these tables,
including shape-manipulation, array creation, and iteration routines.

It's assumed that you've imported NumPy like this:

```python
import numpy as np
```

and `ndarray` like this:

```rust
#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
```

### Array creation

This table contains ways to create arrays from scratch. For creating arrays by
operations on other arrays (e.g. arithmetic), see the other tables. Also see
the [`::from_vec()`][::from_vec()], [`::from_iter()`][::from_iter()],
[`::default()`][::default()], [`::from_shape_fn()`][::from_shape_fn()], and
[`::from_shape_vec_unchecked()`][::from_shape_vec_unchecked()] methods.

NumPy | `ndarray` | Notes
------|-----------|------
`np.array([[1.,2.,3.], [4.,5.,6.]])` | [`array![[1.,2.,3.], [4.,5.,6.]]`][array!] or [`arr2(&[[1.,2.,3.], [4.,5.,6.]])`][arr2()] | 2×3 floating-point array literal
`np.arange(0., 10., 0.5)` or `np.r_[:10.:0.5]` | [`Array::range(0., 10., 0.5)`][::range()] | create a 1-D array with values `0.`, `0.5`, …, `9.5`
`np.linspace(0., 10., 11)` or `np.r_[:10.:11j]` | [`Array::linspace(0., 10., 11)`][::linspace()] | create a 1-D array with 11 elements with values `0.`, …, `10.`
`np.zeros((3, 4, 5))` | [`Array::zeros((3, 4, 5))`][::zeros()] | create a 3×4×5 array filled with zeros (inferring the element type)
`np.zeros((3, 4, 5), order='F')` | [`Array::zeros((3, 4, 5).f())`][::zeros()] | create a 3×4×5 array with Fortran (column-major) memory layout filled with zeros (inferring the element type)
`np.full((3, 4), 7.)` | [`Array::from_elem((3, 4), 7.)`][::from_elem()] | create a 3×4 array filled with the value `7.`
`np.eye(3)` | [`Array::eye(3)`][::eye()] | create a 3×3 identity matrix (inferring the element type)
`np.array([1, 2, 3, 4]).reshape((2, 2))` | [`Array::from_shape_vec((2, 2), vec![1, 2, 3, 4])?`][::from_shape_vec()] | create a 2×2 array from the elements in the list/`Vec`
`np.array([1, 2, 3, 4]).reshape((2, 2), order='F')` | [`Array::from_shape_vec((2, 2).f(), vec![1, 2, 3, 4])?`][::from_shape_vec()] | create a 2×2 array from the elements in the list/`Vec` using Fortran (column-major) order
`np.empty((3, 4))` | [`unsafe { Array::uninitialized((3, 4)) }`][::uninitialized()] | create a 3×4 uninitialized array (inferring the element type)
`np.random` | See the [`ndarray-rand`](https://crates.io/crates/ndarray-rand) crate. | create arrays of random numbers

### Indexing and slicing

A few notes:

* Indices start at 0. For example, "row 1" is the second row in the array.

* Some methods have multiple variants in terms of ownership and mutability.
  Only the non-mutable methods that take the array by reference are listed in
  this table. For example, [`.slice()`][.slice()] also has corresponding
  methods [`.slice_mut()`][.slice_mut()], [`.slice_move()`][.slice_move()], and
  [`.slice_inplace()`][.slice_inplace()].

* The behavior of slicing is slightly different from NumPy for slices with
  `step < -1`. See the docs for the [`s![]` macro][s!] for more details.

NumPy | `ndarray` | Notes
------|-----------|------
`a[-1]` | [`a[a.len()-1]`][.index()] | access the last element in 1-D array `a`
`a[1,4]` | [`a[(1,4)]`][.index()] | access the element in row 1, column 4
`a[1]` or `a[1,:,:]` | [`a.slice(s![1, .., ..])`][.slice()] or [`a.subview(Axis(0), 1)`][.subview()] | get a 2-D subview of a 3-D array at index 1 of axis 0
`a[0:5]` or `a[:5]` or `a[0:5,:]` | [`a.slice(s![0..5, ..])`][.slice()] or [`a.slice(s![..5, ..])`][.slice()] or [`a.slice_axis(Axis(0), (0..5).into())`][.slice_axis()] | get the first 5 rows of a 2-D array
`a[-5:]` or `a[-5:,:]` | [`a.slice(s![-5.., ..])`][.slice()] or [`a.slice_axis(Axis(0), (-5..).into())`][.slice_axis()] | get the last 5 rows of a 2-D array
`a[:3,4:9]` | [`a.slice(s![..3, 4..9])`][.slice()] | columns 4, 5, 6, 7, and 8 of the first 3 rows
`a[1:4:2,::-1]` | [`a.slice(s![1..4;2, ..;-1])`][.slice()] | rows 1 and 3 with the columns in reverse order

### Shape and strides

NumPy | `ndarray` | Notes
------|-----------|------
`np.ndim(a)` or `a.ndim` | [`a.ndim()`][.ndim()] | get the number of dimensions of array `a`
`np.size(a)` or `a.size` | [`a.len()`][.len()] | get the number of elements in array `a`
`np.shape(a)` or `a.shape` | [`a.shape()`][.shape()] or [`a.dim()`][.dim()] | get the shape of array `a`
`a.shape[axis]` | [`a.len_of(Axis(axis))`][.len_of()] or `a.shape()[axis]` | get the length of an axis
`a.strides` | [`a.strides()`][.strides()] | get the strides of array `a`
`np.size(a) == 0` or `a.size == 0` | [`a.is_empty()`][.is_empty()] | check if the array has zero elements

### Mathematics

Note that [`.mapv()`][.mapv()] has corresponding methods [`.map()`][.map()],
[`.mapv_into()`][.mapv_into()], [`.map_inplace()`][.map_inplace()], and
[`.mapv_inplace()`][.mapv_inplace()]. Also look at [`.fold()`][.fold()],
[`.visit()`][.visit()], [`.fold_axis()`][.fold_axis()], and
[`.map_axis()`][.map_axis()].

<table>
<tr><th>

NumPy

</th><th>

`ndarray`

</th><th>

Notes

</th></tr>

<tr><td>

`a.transpose()` or `a.T`

</td><td>

[`a.t()`][.t()] or [`a.reversed_axes()`][.reversed_axes()]

</td><td>

transpose of array `a`

</td></tr>

<tr><td>

`mat1.dot(mat2)`

</td><td>

[`mat1.dot(&mat2)`][matrix-* dot]

</td><td>

2-D matrix multiply

</td></tr>

<tr><td>

`mat.dot(vec)`

</td><td>

[`mat.dot(&vec)`][matrix-* dot]

</td><td>

2-D matrix dot 1-D column vector

</td></tr>

<tr><td>

`vec.dot(mat)`

</td><td>

[`vec.dot(&mat)`][vec-* dot]

</td><td>

1-D row vector dot 2-D matrix

</td></tr>

<tr><td>

`vec1.dot(vec2)`

</td><td>

[`vec1.dot(&vec2)`][vec-* dot]

</td><td>

vector dot product

</td></tr>

<tr><td>

`a * b`, `a + b`, etc.

</td><td>

[`a * b`, `a + b`, etc.](https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#arithmetic-operations)

</td><td>

element-wise arithmetic operations

</td></tr>

<tr><td>

`a**3`

</td><td>

[`a.mapv(|a| a.powi(3))`][.mapv()]

</td><td>

element-wise power of 3

</td></tr>

<tr><td>

`np.sqrt(a)`

</td><td>

[`a.mapv(f64::sqrt)`][.mapv()]

</td><td>

element-wise square root for `f64` array

</td></tr>

<tr><td>

`(a>0.5)`

</td><td>

[`a.mapv(|a| a > 0.5)`][.mapv()]

</td><td>

array of `bool`s of same shape as `a` with `true` where `a > 0.5` and `false` elsewhere

</td></tr>

<tr><td>

`np.sum(a)` or `a.sum()`

</td><td>

[`a.scalar_sum()`][.scalar_sum()]

</td><td>

sum the elements in `a`

</td></tr>

<tr><td>

`np.sum(a, axis=2)` or `a.sum(axis=2)`

</td><td>

[`a.sum_axis(Axis(2))`][.sum_axis()]

</td><td>

sum the elements in `a` along axis 2

</td></tr>

<tr><td>

`np.mean(a)` or `a.mean()`

</td><td>

`a.scalar_sum() / a.len() as f64`

</td><td>

calculate the mean of the elements in `f64` array `a`

</td></tr>

<tr><td>

`np.mean(a, axis=2)` or `a.mean(axis=2)`

</td><td>

[`a.mean_axis(Axis(2))`][.mean_axis()]

</td><td>

calculate the mean of the elements in `a` along axis 2

</td></tr>

<tr><td>

`np.allclose(a, b, atol=1e-8)`

</td><td>

[`a.all_close(&b, 1e-8)`][.all_close()]

</td><td>

check if the arrays' elementwise differences are within an absolute tolerance

</td></tr>

<tr><td>

`np.diag(a)`

</td><td>

[`a.diag()`][.diag()]

</td><td>

view the diagonal of `a`

</td></tr>

<tr><td>

`np.linalg`

</td><td>

See other crates, e.g.
[`ndarray-linalg`](https://crates.io/crates/ndarray-linalg) and
[`linxal`](https://crates.io/crates/linxal).

</td><td>

linear algebra (matrix inverse, solving, decompositions, etc.)

</td></tr>
</table>

### Array manipulation

NumPy | `ndarray` | Notes
------|-----------|------
`a[:] = 3.` | [`a.fill(3.)`][.fill()] | set all array elements to the same scalar value
`a[:] = b` | [`a.assign(&b)`][.assign()] | copy the data from array `b` into array `a`
`np.concatenate((a,b), axis=1)` | [`stack![Axis(1), a, b]`][stack!] or [`stack(Axis(1), &[a.view(), b.view()])`][stack()] | concatenate arrays `a` and `b` along axis 1
`a[:,np.newaxis]` or `np.expand_dims(a, axis=1)` | [`a.insert_axis(Axis(1))`][.insert_axis()] | create an array from `a`, inserting a new axis 1
`a.transpose()` or `a.T` | [`a.t()`][.t()] or [`a.reversed_axes()`][.reversed_axes()] | transpose of array `a`
`np.diag(a)` | [`a.diag()`][.diag()] | view the diagonal of `a`
`a.flat` | [`a.iter()`][.iter()] | iterator over the array elements in logical order
`a.flatten()` | [`Array::from_iter(a.iter())`][::from_iter()] | create a 1-D array by flattening `a`

### Convenience methods for 2-D arrays

NumPy | `ndarray` | Notes
------|-----------|------
`len(a)` or `a.shape[0]` | [`a.rows()`][.rows()] | get the number of rows in a 2-D array
`a.shape[1]` | [`a.cols()`][.cols()] | get the number of columns in a 2-D array
`a[1]` or `a[1,:]` | [`a.row(1)`][.row()] or [`a.row_mut(1)`][.row_mut()] | view (or mutable view) of row 1 in a 2-D array
`a[:,4]` | [`a.column(4)`][.column()] or [`a.column_mut(4)`][.column_mut()] | view (or mutable view) of column 4 in a 2-D array
`a.shape[0] == a.shape[1]` | [`a.is_square()`][.is_square()] | check if the array is square

[.all_close()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.all_close
[array!]: https://docs.rs/ndarray/0.11/ndarray/macro.array.html
[arr2()]: https://docs.rs/ndarray/0.11/ndarray/fn.arr2.html
[.assign()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.assign
[.cols()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.cols
[.column()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.column
[.column_mut()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.column_mut
[::default()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.default
[.diag()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.diag
[.dim()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.dim
[::eye()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.eye
[.fill()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.fill
[.fold()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.fold
[.fold_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.fold_axis
[::from_elem()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_elem
[::from_iter()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_iter
[::from_shape_fn()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_shape_fn
[::from_shape_vec()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_shape_vec
[::from_shape_vec_unchecked()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_shape_vec_unchecked
[::from_vec()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_vec
[.index()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#impl-Index<I>
[.insert_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.insert_axis
[.is_empty()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.is_empty
[.is_square()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.is_square
[.iter()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.iter
[.len()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.len
[.len_of()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.len_of
[::linspace()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.linspace
[.map()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.map
[.map_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.map_axis
[.map_inplace()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.map_inplace
[.mapv()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.mapv
[.mapv_inplace()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.mapv_inplace
[.mapv_into()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.mapv_into
[matrix-* dot]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.dot-1
[.mean_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.mean_axis
[.ndim()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.ndim
[::range()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.range
[.reversed_axes()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.reversed_axes
[.row()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.row
[.row_mut()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.row_mut
[.rows()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.rows
[s!]: https://docs.rs/ndarray/0.11/ndarray/macro.s.html
[.scalar_sum()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.scalar_sum
[.slice()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice
[.slice_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice_axis
[.slice_inplace()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice_inplace
[.slice_move()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice_move
[.slice_mut()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice_mut
[.shape()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.shape
[stack!]: https://docs.rs/ndarray/0.11/ndarray/macro.stack.html
[stack()]: https://docs.rs/ndarray/0.11/ndarray/fn.stack.html
[.strides()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.strides
[.subview()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.subview
[.sum_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.sum_axis
[.t()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.t
[::uninitialized()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.uninitialized
[vec-* dot]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.dot
[.visit()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.visit
[::zeros()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.zeros
