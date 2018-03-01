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

When slicing in `ndarray`, the axis is first sliced with `start..end`. Then
if `step` is positive, the first index is the front of the slice; if `step`
is negative, the first index is the back of the slice. This means that the
behavior is the same as NumPy except when `step < -1`. See the docs for the
[`s![]` macro](macro.s.html) for more details.

</td>
</tr>
</table>

## Rough `ndarray`–NumPy equivalents

A few notes about this table:

* Indices start at 0. For example, "row 1" is the second row in the array.

* Some methods have multiple variants in terms of ownership and mutability.
  Only the non-mutable methods that take the array by reference are listed.
  For example, `.slice()` also has corresponding methods `.slice_mut()`,
  `.slice_move()`, and `.slice_inplace()`.

* There are some convenience methods for 2-D arrays that are not included
  in this table. See the table below this one.

* There are a variety of other methods that aren't included in this table,
  including shape-manipulation, array creation, and iteration routines.

* It's assumed that you've imported NumPy like this:

  ```python
  import numpy as np
  ```

  and `ndarray` like this:

  ```rust
  #[macro_use]
  extern crate ndarray;

  use ndarray::prelude::*;
  ```

<table>
<tr><th>

NumPy

</th><th>

`ndarray`

</th><th>

Notes

</th></tr>

<tr><td>

`np.ndim(a)` or `a.ndim`

</td><td>

[`a.ndim()`][.ndim()]

</td><td>

get the number of dimensions of array `a`

</td></tr>

<tr><td>

`np.size(a)` or `a.size`

</td><td>

[`a.len()`][.len()]

</td><td>

get the number of elements in array `a`

</td></tr>

<tr><td>

`np.shape(a)` or `a.shape`

</td><td>

[`a.shape()`][.shape()] or [`a.dim()`][.dim()]

</td><td>

get the shape of array `a`

</td></tr>

<tr><td>

`a.shape[axis]`

</td><td>

[`a.len_of(axis)`][.len_of()] or `a.shape()[axis]`

</td><td>

get the length of an axis

</td></tr>

<tr><td>

`np.array([[1.,2.,3.], [4.,5.,6.]])`

</td><td>

[`array![[1.,2.,3.], [4.,5.,6.]]`][array!] or [`arr2(&[[1.,2.,3.], [4.,5.,6.]])`][arr2()]

</td><td>

2×3 floating-point array literal

</td></tr>

<tr><td>

`np.concatenate((a,b), axis=1)`

</td><td>

[`stack![Axis(1), a, b]`][stack!] or [`stack(Axis(1), &[a.view(), b.view()])`][stack()]

</td><td>

concatenate arrays `a` and `b` along axis 1

</td></tr>

<tr><td>

`a[-1]`

</td><td>

[`a[a.len()-1]`][.index()]

</td><td>

access the last element in 1-D array `a`

</td></tr>

<tr><td>

`a[1,4]`

</td><td>

[`a[(1,4)]`][.index()]

</td><td>

access the element in row 1, column 4

</td></tr>

<tr><td>

`a[1]` or `a[1,:,:]`

</td><td>

[`a.slice(s![1, .., ..])`][.slice()] or [`a.subview(Axis(0), 1)`][.subview()]

</td><td>

get a 2-D subview of a 3-D array at index 1 of axis 0

</td></tr>

<tr><td>

`a[0:5]` or `a[:5]` or `a[0:5,:]`

</td><td>

[`a.slice(s![0..5, ..])`][.slice()] or [`a.slice(s![..5, ..])`][.slice()] or [`a.slice_axis(Axis(0), (0..5).into())`][.slice_axis()]

</td><td>

get the first 5 rows of a 2-D array

</td></tr>

<tr><td>

`a[-5:]` or `a[-5:,:]`

</td><td>

[`a.slice(s![-5.., ..])`][.slice()] or [`a.slice_axis(Axis(0), (-5..).into())`][.slice_axis()]

</td><td>

get the last 5 rows of a 2-D array

</td></tr>

<tr><td>

`a[:3,4:9]`

</td><td>

[`a.slice(s![..3, 4..9])`][.slice()]

</td><td>

columns 4, 5, 6, 7, and 8 of the first 3 rows

</td></tr>

<tr><td>

`a[1:4:2,::-1]`

</td><td>

[`a.slice(s![1..4;2, ..;-1])`][.slice()]

</td><td>

rows 1 and 3 with the columns in reverse order

</td></tr>

<tr><td>

`np.expand_dims(a, axis=1)`

</td><td>

[`a.insert_axis(Axis(1))`][.insert_axis()]

</td><td>

create an array from `a`, inserting a new axis 1

</td></tr>

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

`a[:] = 3.`

</td><td>

[`a.fill(3.)`][.fill()]

</td><td>

set all array elements to the same scalar value

</td></tr>

<tr><td>

`a.flat`

</td><td>

[`a.iter()`][.iter()]

</td><td>

iterator over the array elements in logical order

</td></tr>

<tr><td>

`a.flatten()`

</td><td>

[`Array::from_iter(a.iter())`][::from_iter()]

</td><td>

create a 1-D array by flattening `a`

</td></tr>

<tr><td>

`np.arange(0., 10., 0.5)` or `np.r_[:10.:0.5]`

</td><td>

[`Array::range(0., 10., 0.5)`][::range()]

</td><td>

create a 1-D array with values `0.`, `0.5`, …, `9.5`

</td></tr>

<tr><td>

`np.linspace(0., 10., 11)` or `np.r_[:10.:11j]`

</td><td>

[`Array::linspace(0., 10., 11)`][::linspace()]

</td><td>

create a 1-D array with 11 elements with values `0.`, …, `10.`

</td></tr>

<tr><td>

`np.zeros((3, 4, 5))`

</td><td>

[`Array::zeros((3, 4, 5))`][::zeros()]

</td><td>

create a 3×4×5 array filled with zeros (inferring the element type)

</td></tr>

<tr><td>

`np.full((3, 4), 7.)`

</td><td>

[`Array::from_elem((3, 4), 7.)`][::from_elem()]

</td><td>

create a 3×4 array filled with the value `7.`

</td></tr>

<tr><td>

`np.eye(3)`

</td><td>

[`Array::eye(3)`][::eye()]

</td><td>

create a 3×3 identity matrix (inferring the element type)

</td></tr>

<tr><td>

`np.diag(a)`

</td><td>

[`a.diag()`][.diag()]

</td><td>

view the diagonal of `a`

</td></tr>

<tr><td>

`np.random`

</td><td>

See the [`ndarray-rand`](https://crates.io/crates/ndarray-rand) crate.

</td><td>

create arrays of random numbers

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

[.ndim()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.ndim
[.len()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.len
[.shape()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.shape
[.dim()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.dim
[.len_of()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.len_of
[array!]: https://docs.rs/ndarray/0.11/ndarray/macro.array.html
[arr2()]: https://docs.rs/ndarray/0.11/ndarray/fn.arr2.html
[stack!]: https://docs.rs/ndarray/0.11/ndarray/macro.stack.html
[stack()]: https://docs.rs/ndarray/0.11/ndarray/fn.stack.html
[.index()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#impl-Index<I>
[.slice()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice
[.insert_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.insert_axis
[.subview()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.subview
[.slice_axis()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.slice_axis
[.t()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.t
[.reversed_axes()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.reversed_axes
[matrix-* dot]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.dot-1
[vec-* dot]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.dot
[.mapv()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.mapv
[.fill()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.fill
[.iter()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.iter
[::from_iter()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_iter
[::range()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.range
[::linspace()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.linspace
[::zeros()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.zeros
[::from_elem()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.from_elem
[::eye()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.eye
[.diag()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.diag

There are some convenience methods for 2-D arrays:

NumPy | `ndarray` | Notes
------|-----------|------
`len(a)` or `a.shape[0]` | [`a.rows()`][.rows()] | get the number of rows in a 2-D array
`a.shape[1]` | [`a.cols()`][.cols()] | get the number of columns in a 2-D array
`a[1]` or `a[1,:]` | [`a.row(1)`][.row()] or [`a.row_mut(1)`][.row_mut()] | view (or mutable view) of row 1 in a 2-D array
`a[:,4]` | [`a.column(4)`][.column()] or [`a.column_mut(4)`][.column_mut()] | view (or mutable view) of column 4 in a 2-D array
`a.shape[0] == a.shape[1]` | [`a.is_square()`][.is_square()] | check if the array is square

[.rows()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.rows
[.cols()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.cols
[.row()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.row
[.row_mut()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.row_mut
[.column()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.column
[.column_mut()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.column_mut
[.is_square()]: https://docs.rs/ndarray/0.11/ndarray/struct.ArrayBase.html#method.is_square
