use {Array, ArrayBase, ArrayView, ArrayViewMut, Data, DataMut, DataOwned, Dimension,
     IntoDimension, ShapeError};

pub enum ArrayViewTemp<'a, A: 'a, D: Dimension> {
    Temporary(Array<A, D>),
    View(ArrayView<'a, A, D>),
}

impl<'a, A, D> ArrayViewTemp<'a, A, D>
where
    D: Dimension,
{
    pub fn view(&self) -> ArrayView<A, D> {
        match *self {
            ArrayViewTemp::Temporary(ref tmp) => tmp.view(),
            ArrayViewTemp::View(ref view) => view.view(),
        }
    }
}

pub struct ArrayViewMutTempRepr<'a, A, Do, Dt>
where
    A: 'a,
    Do: Dimension,
    Dt: Dimension,
{
    /// Mutable view of the original array.
    view: ArrayViewMut<'a, A, Do>,
    /// Temporary owned array that can be modified through `.view_mut()`.
    tmp: Array<A, Dt>,
    /// Closure that gets called `(drop_hook)(view, tmp)` when `self` is dropped.
    drop_hook: Box<FnMut(&mut ArrayViewMut<'a, A, Do>, &mut Array<A, Dt>)>,
}

impl<'a, A, Do, Dt> ArrayViewMutTempRepr<'a, A, Do, Dt>
where
    Do: Dimension,
    Dt: Dimension,
{
    pub fn new(
        view: ArrayViewMut<'a, A, Do>,
        tmp: Array<A, Dt>,
        drop_hook: Box<FnMut(&mut ArrayViewMut<'a, A, Do>, &mut Array<A, Dt>)>,
    ) -> Self {
        ArrayViewMutTempRepr {
            view,
            tmp,
            drop_hook,
        }
    }

    /// Returns a view of the temporary array.
    pub fn view(&self) -> ArrayView<A, Dt> {
        self.tmp.view()
    }

    /// Returns a mutable view of the temporary array.
    pub fn view_mut(&mut self) -> ArrayViewMut<A, Dt> {
        self.tmp.view_mut()
    }
}

impl<'a, A, Do, Dt> Drop for ArrayViewMutTempRepr<'a, A, Do, Dt>
where
    Do: Dimension,
    Dt: Dimension,
{
    fn drop(&mut self) {
        let ArrayViewMutTempRepr {
            ref mut view,
            ref mut tmp,
            ref mut drop_hook,
        } = *self;
        (drop_hook)(view, tmp);
    }
}

pub enum ArrayViewMutTemp<'a, A, Do, Dt>
where
    A: 'a,
    Do: Dimension,
    Dt: Dimension,
{
    Temporary(ArrayViewMutTempRepr<'a, A, Do, Dt>),
    View(ArrayViewMut<'a, A, Dt>),
}

impl<'a, A, Do, Dt> ArrayViewMutTemp<'a, A, Do, Dt>
where
    Do: Dimension,
    Dt: Dimension,
{
    pub fn view(&self) -> ArrayView<A, Dt> {
        match *self {
            ArrayViewMutTemp::Temporary(ref tmp) => tmp.view(),
            ArrayViewMutTemp::View(ref view) => view.view(),
        }
    }

    pub fn view_mut(&mut self) -> ArrayViewMut<A, Dt> {
        match *self {
            ArrayViewMutTemp::Temporary(ref mut tmp) => tmp.view_mut(),
            ArrayViewMutTemp::View(ref mut view) => view.view_mut(),
        }
    }
}

/// This is analogous to the `order` parameter in
/// [`numpy.reshape()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html#numpy.reshape).
pub enum Order {
    /// C-like order
    RowMajor,
    /// Fortran-like order
    ColMajor,
    /// Fortran-like order if the array is Fortran contiguous in memory, C-like order otherwise
    Automatic,
}

impl<A, S, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Returns an `ArrayViewTemp` instance with the desired shape.
    ///
    /// The reshaped data can be read by calling `.view()` on the
    /// `ArrayViewTemp` instance.
    ///
    /// This method does not require the data to be contiguous in memory.
    ///
    /// **Errors** if `self` doesn't have the same number of elements as `shape`.
    pub fn view_with_shape<E>(
        &self,
        shape: E,
        order: Order,
    ) -> Result<ArrayViewTemp<A, E::Dim>, ShapeError>
    where
        A: Clone,
        E: IntoDimension,
    {
        match order {
            Order::RowMajor => if self.is_standard_layout() {
                Ok(ArrayViewTemp::View(self.view().into_shape(shape)?))
            } else {
                let tmp = Array::from_iter(self.iter().cloned()).into_shape(shape)?;
                Ok(ArrayViewTemp::Temporary(tmp))
            },
            Order::ColMajor => unimplemented!(),
            Order::Automatic => {
                if self.ndim() > 1 && self.view().reversed_axes().is_standard_layout() {
                    self.view_with_shape(shape, Order::ColMajor)
                } else {
                    self.view_with_shape(shape, Order::RowMajor)
                }
            }
        }
    }

    /// Returns an `ArrayViewMutTemp` instance with the desired shape.
    ///
    /// The reshaped data can be read/written by calling `.view_mut()` on the
    /// `ArrayViewMutTemp` instance.
    ///
    /// This method does not require the data to be contiguous in memory.
    ///
    /// **Errors** if `self` doesn't have the same number of elements as `shape`.
    pub fn view_mut_with_shape<E>(
        &mut self,
        shape: E,
        order: Order,
    ) -> Result<ArrayViewMutTemp<A, D, E::Dim>, ShapeError>
    where
        A: Clone,
        S: DataMut,
        E: IntoDimension,
    {
        match order {
            Order::RowMajor => if self.is_standard_layout() {
                Ok(ArrayViewMutTemp::View(self.view_mut().into_shape(shape)?))
            } else {
                let tmp = Array::from_iter(self.iter().cloned()).into_shape(shape)?;
                Ok(ArrayViewMutTemp::Temporary(ArrayViewMutTempRepr::new(
                    self.view_mut(),
                    tmp,
                    Box::new(|view, tmp| {
                        view.iter_mut()
                            .zip(tmp.iter())
                            .for_each(|(o, t)| *o = t.clone())
                    }),
                )))
            },
            Order::ColMajor => unimplemented!(),
            Order::Automatic => {
                if self.ndim() > 1 && self.view().reversed_axes().is_standard_layout() {
                    self.view_mut_with_shape(shape, Order::ColMajor)
                } else {
                    self.view_mut_with_shape(shape, Order::RowMajor)
                }
            }
        }
    }

    /// Returns a new array with the desired shape.
    ///
    /// This method does not require the data to be contiguous in memory.
    ///
    /// **Errors** if `self` doesn't have the same number of elements as `shape`.
    pub fn into_shape_owned<E>(
        self,
        shape: E,
        order: Order,
    ) -> Result<ArrayBase<S, E::Dim>, ShapeError>
    where
        A: Clone,
        S: DataOwned,
        E: IntoDimension,
    {
        match order {
            Order::RowMajor => if self.is_standard_layout() {
                self.into_shape(shape)
            } else {
                unimplemented!()
            },
            Order::ColMajor => unimplemented!(),
            Order::Automatic => {
                if self.ndim() > 1 && self.view().reversed_axes().is_standard_layout() {
                    self.into_shape_owned(shape, Order::ColMajor)
                } else {
                    self.into_shape_owned(shape, Order::RowMajor)
                }
            }
        }
    }
}
