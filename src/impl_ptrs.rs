use imp_prelude::*;

impl<A, D> ArrayPtr<A, D>
where
    D: Dimension,
{
    /// Return a read-only view of the array.
    ///
    /// **Warning** this is equivalent to dereferencing a raw pointer; you must
    /// choose the correct lifetime.
    pub unsafe fn view<'a>(&self) -> ArrayView<'a, A, D> {
        ArrayView::new_(self.ptr, self.dim.clone(), self.strides.clone())
    }
}

impl<A, D> ArrayMutPtr<A, D>
where
    D: Dimension,
{
    /// Return a read-only view of the array
    ///
    /// **Warning** this is equivalent to dereferencing a raw pointer; you must
    /// choose the correct lifetime.
    pub unsafe fn view<'a>(&self) -> ArrayView<'a, A, D> {
        ArrayView::new_(self.ptr, self.dim.clone(), self.strides.clone())
    }

    /// Return a read-write view of the array
    ///
    /// **Warning** this is equivalent to dereferencing a raw pointer; you must
    /// choose the correct lifetime.
    pub unsafe fn view_mut<'a>(&mut self) -> ArrayViewMut<'a, A, D> {
        ArrayViewMut::new_(self.ptr, self.dim.clone(), self.strides.clone())
    }
}
