use num_traits::{float::Float, real::Real, NumAssign, Signed};

use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

// #[cfg(test)]
use std::fmt::Debug;

use paste::paste;

#[derive(Debug, Clone, Copy)]
pub struct Matrix<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>(
    pub [[T; COL]; ROW],
);

/// get a 3x3 subregion of the input matrix
impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
    From<(Matrix<T, ROW, COL>, [Range<usize>; 2])> for Matrix<T, 3, 3>
{
    fn from((m, [range_rows, range_cols]): (Matrix<T, ROW, COL>, [Range<usize>; 2])) -> Self {
        let mut ret = Self::zero();
        assert!(range_rows.end <= ROW);
        assert!(range_cols.end <= COL);
        assert!(range_rows.end - range_rows.start <= 3);
        assert!(range_cols.end - range_cols.start <= 3);
        for (out_i, i) in (range_rows.start..range_rows.end).enumerate() {
            for (out_j, j) in (range_cols.start..range_cols.end).enumerate() {
                ret[[out_i, out_j]] = m[[i, j]];
            }
        }
        ret
    }
}

#[derive(Debug, Clone)]
pub(crate) enum RangeType {
    Range(Range<usize>),
    RangeFrom(RangeFrom<usize>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<usize>),
    RangeTo(RangeTo<usize>),
    RangeToInclusive(RangeToInclusive<usize>),
}

pub struct MatrixSliceMut<'a, T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> {
    pub matrix: &'a mut Matrix<T, ROW, COL>,
    range: (RangeType, RangeType),
}

pub struct MatrixSlice<'a, T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> {
    pub matrix: &'a Matrix<T, ROW, COL>,
    range: (RangeType, RangeType),
}

impl<'a, T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
    From<MatrixSliceMut<'a, T, ROW, COL>> for MatrixSlice<'a, T, ROW, COL>
{
    fn from(m: MatrixSliceMut<'a, T, ROW, COL>) -> Self {
        Self {
            matrix: m.matrix,
            range: m.range,
        }
    }
}

macro_rules! impl_matrix_slice_mut {
    ($type_range_rows:ty, $type_range_cols:ty, $enum_range_rows:ty,  $enum_range_cols:ty) => {
        paste! {
            /// get a mutable slice of the input matrix
            impl<'a, T: 'a + NumAssign + Copy + Default, const ROW: usize, const COL: usize>
                From<(
                    &'a mut Matrix<T, ROW, COL>,
                    ($type_range_rows, $type_range_cols),
                )> for MatrixSliceMut<'a, T, ROW, COL>
            {
                fn from(
                    (m, (range_rows, range_cols)): (
                        &'a mut Matrix<T, ROW, COL>,
                        ($type_range_rows, $type_range_cols),
                    ),
                ) -> Self {
                    Self {
                        matrix: m,
                        range: ($enum_range_rows(range_rows), $enum_range_cols(range_cols)),
                    }
                }
            }
            /// get a slice of the input matrix
            ///
            /// ```compile_fail
            /// let (m1, mut m2) = (
            ///     Matrix::<f64, 15, 10>::default(),
            ///     Matrix::<f64, 15, 15>::default(),
            /// );
            /// let src = MatrixSlice::from((&m1, [(2..15), (3..15)]));
            /// let mut dest = MatrixSliceMut::from((&mut m2, [(5..8), (7..10)]));
            /// drop(m2);
            /// dest.assign(src);
            impl<'a, T: 'a + NumAssign + Copy + Default, const ROW: usize, const COL: usize>
                From<(
                    &'a Matrix<T, ROW, COL>,
                    ($type_range_rows, $type_range_cols),
                )> for MatrixSlice<'a, T, ROW, COL>
            {
                fn from(
                    (m, (range_rows, range_cols)): (
                        &'a Matrix<T, ROW, COL>,
                        ($type_range_rows, $type_range_cols),
                    ),
                ) -> Self {
                    Self {
                        matrix: m,
                        range: ($enum_range_rows(range_rows), $enum_range_cols(range_cols)),
                    }
                }
            }
        }
    };
}

macro_rules! impl_matrix_slice_outer {
    ($type_range_rows:ty, $enum_range_rows:ty) => {
        impl_matrix_slice_mut!(
            $type_range_rows,
            Range<usize>,
            $enum_range_rows,
            RangeType::Range
        );
        impl_matrix_slice_mut!(
            $type_range_rows,
            RangeFrom<usize>,
            $enum_range_rows,
            RangeType::RangeFrom
        );
        impl_matrix_slice_mut!(
            $type_range_rows,
            RangeFull,
            $enum_range_rows,
            RangeType::RangeFull
        );
        impl_matrix_slice_mut!(
            $type_range_rows,
            RangeTo<usize>,
            $enum_range_rows,
            RangeType::RangeTo
        );
        impl_matrix_slice_mut!(
            $type_range_rows,
            RangeToInclusive<usize>,
            $enum_range_rows,
            RangeType::RangeToInclusive
        );
        impl_matrix_slice_mut!(
            $type_range_rows,
            RangeInclusive<usize>,
            $enum_range_rows,
            RangeType::RangeInclusive
        );
    };
}

impl_matrix_slice_outer!(Range<usize>, RangeType::Range);
impl_matrix_slice_outer!(RangeFull, RangeType::RangeFull);
impl_matrix_slice_outer!(RangeFrom<usize>, RangeType::RangeFrom);
impl_matrix_slice_outer!(RangeTo<usize>, RangeType::RangeTo);
impl_matrix_slice_outer!(RangeToInclusive<usize>, RangeType::RangeToInclusive);
impl_matrix_slice_outer!(RangeInclusive<usize>, RangeType::RangeInclusive);

#[derive(Default)]
struct RangeInfo {
    pub start: usize,
    pub count: usize,
}

/// # Copy values from one slice to another.
/// # Example
/// ```
/// use lightmatrix::matrix::*;
/// let (mut m1, mut m2) = (
///     Matrix::<f64, 15, 10>::default(),
///     Matrix::<f64, 15, 15>::default(),
/// );
///
/// //fill matrices with values...
///
/// let src = MatrixSlice::from((&m1, [(2..15), (3..15)]));
/// let mut dest = MatrixSliceMut::from((&mut m2, [(5..8), (7..10)]));
/// dest.assign(src);
/// ```
impl<'a, T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
    MatrixSliceMut<'a, T, ROW, COL>
{
    pub fn assign<const ROW_OTHER: usize, const COL_OTHER: usize>(
        &mut self,
        other: MatrixSlice<'a, T, ROW_OTHER, COL_OTHER>,
    ) {
        let (range_rows, range_cols) = self.get_range();
        let dest_range_rows_count = range_rows.count;
        let dest_range_cols_count = range_cols.count;
        let dest_rows_start = range_rows.start;
        let dest_cols_start = range_cols.start;

        let (range_rows, range_cols) = other.get_range();
        let src_range_rows_count = range_rows.count;
        let src_range_cols_count = range_cols.count;
        let src_rows_start = range_rows.start;
        let src_cols_start = range_cols.start;

        assert!(dest_range_rows_count <= src_range_rows_count);

        assert!(dest_range_cols_count <= src_range_cols_count);

        for i in 0..dest_range_rows_count {
            for j in 0..dest_range_cols_count {
                self.matrix[[dest_rows_start + i, dest_cols_start + j]] =
                    other.matrix[[src_rows_start + i, src_cols_start + j]];
            }
        }
    }
    fn get_range(&'a self) -> (RangeInfo, RangeInfo) {
        let mut ranges: [RangeInfo; 2] = Default::default();
        for (idx, r) in (&[&self.range.0, &self.range.1]).iter().enumerate() {
            let range_count: usize;
            let range_start: usize;
            match r {
                RangeType::Range(Range { start, end }) => {
                    range_count = end - start;
                    range_start = *start;
                }
                RangeType::RangeInclusive(ref range_inclusive @ _) => {
                    let (&start, &end) = (range_inclusive.start(), range_inclusive.end());
                    range_count = end - start + 1;
                    range_start = start;
                }
                RangeType::RangeFull(_) => {
                    if idx == 0 {
                        range_count = self.matrix.rows();
                    } else {
                        range_count = self.matrix.cols();
                    }
                    range_start = 0;
                }
                RangeType::RangeFrom(RangeFrom { start }) => {
                    if idx == 0 {
                        range_count = self.matrix.rows();
                    } else {
                        range_count = self.matrix.cols();
                    }
                    range_start = *start;
                }
                RangeType::RangeTo(RangeTo { end }) => {
                    range_count = *end;
                    range_start = 0;
                }
                RangeType::RangeToInclusive(RangeToInclusive { end }) => {
                    range_count = end + 1;
                    range_start = 0;
                }
            }
            ranges[idx].start = range_start;
            ranges[idx].count = range_count;
        }
        let [range_rows, range_cols] = ranges;
        (range_rows, range_cols)
    }
}

impl<'a, T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
    MatrixSlice<'a, T, ROW, COL>
{
    fn get_range(&'a self) -> (RangeInfo, RangeInfo) {
        let mut ranges: [RangeInfo; 2] = Default::default();
        for (idx, r) in (&[&self.range.0, &self.range.1]).iter().enumerate() {
            let range_count: usize;
            let range_start: usize;
            match r {
                RangeType::Range(Range { start, end }) => {
                    range_count = end - start;
                    range_start = *start;
                }
                RangeType::RangeInclusive(ref range_inclusive @ _) => {
                    let (&start, &end) = (range_inclusive.start(), range_inclusive.end());
                    range_count = end - start + 1;
                    range_start = start;
                }
                RangeType::RangeFull(_) => {
                    if idx == 0 {
                        range_count = self.matrix.rows();
                    } else {
                        range_count = self.matrix.cols();
                    }
                    range_start = 0;
                }
                RangeType::RangeFrom(RangeFrom { start }) => {
                    if idx == 0 {
                        range_count = self.matrix.rows();
                    } else {
                        range_count = self.matrix.cols();
                    }
                    range_start = *start;
                }
                RangeType::RangeTo(RangeTo { end }) => {
                    range_count = *end;
                    range_start = 0;
                }
                RangeType::RangeToInclusive(RangeToInclusive { end }) => {
                    range_count = end + 1;
                    range_start = 0;
                }
            }
            ranges[idx].start = range_start;
            ranges[idx].count = range_count;
        }
        let [range_rows, range_cols] = ranges;
        (range_rows, range_cols)
    }
}

/// get a 3x1 subregion of the input matrix
impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
    From<(Matrix<T, ROW, COL>, [Range<usize>; 2])> for Matrix<T, 3, 1>
{
    fn from((m, [range_rows, range_cols]): (Matrix<T, ROW, COL>, [Range<usize>; 2])) -> Self {
        let mut ret = Self::zero();
        assert!(range_rows.end <= ROW);
        assert!(range_cols.end <= COL);
        assert!(range_rows.end - range_rows.start <= 3);
        assert!(range_cols.end - range_cols.start <= 1);
        for (out_i, i) in (range_rows.start..range_rows.end).enumerate() {
            for (out_j, j) in (range_cols.start..range_cols.end).enumerate() {
                ret[[out_i, out_j]] = m[[i, j]];
            }
        }
        ret
    }
}

/// get a 1x3 subregion of the input matrix
impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
    From<(Matrix<T, ROW, COL>, [Range<usize>; 2])> for Matrix<T, 1, 3>
{
    fn from((m, [range_rows, range_cols]): (Matrix<T, ROW, COL>, [Range<usize>; 2])) -> Self {
        let mut ret = Self::zero();
        assert!(range_rows.end <= ROW);
        assert!(range_cols.end <= COL);
        assert!(range_rows.end - range_rows.start <= 1);
        assert!(range_cols.end - range_cols.start <= 3);
        for (out_i, i) in (range_rows.start..range_rows.end).enumerate() {
            for (out_j, j) in (range_cols.start..range_cols.end).enumerate() {
                ret[[out_i, out_j]] = m[[i, j]];
            }
        }
        ret
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> Default
    for Matrix<T, ROW, COL>
{
    fn default() -> Self {
        Self([[T::default(); COL]; ROW])
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> Deref
    for Matrix<T, ROW, COL>
{
    type Target = [[T; COL]; ROW];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> DerefMut
    for Matrix<T, ROW, COL>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> Index<[usize; 2]>
    for Matrix<T, ROW, COL>
{
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.0[index[0]][index[1]]
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> IndexMut<[usize; 2]>
    for Matrix<T, ROW, COL>
{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.0[index[0]][index[1]]
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> From<[[T; COL]; ROW]>
    for Matrix<T, ROW, COL>
{
    fn from(x: [[T; COL]; ROW]) -> Self {
        Self(x)
    }
}

macro_rules! impl_partialeq_integral {
    ($type:ty) => {
        impl<const ROW: usize, const COL: usize> PartialEq<Matrix<$type, ROW, COL>>
            for Matrix<$type, ROW, COL>
        {
            fn eq(&self, other: &Self) -> bool {
                for i in 0..ROW {
                    for j in 0..COL {
                        if self.0[i][j] != other.0[i][j] {
                            return false;
                        }
                    }
                }
                true
            }
        }
    };
}

macro_rules! impl_partialeq_float {
    ($type:ty) => {
        impl<const ROW: usize, const COL: usize> PartialEq<Matrix<$type, ROW, COL>>
            for Matrix<$type, ROW, COL>
        {
            fn eq(&self, other: &Self) -> bool {
                for i in 0..ROW {
                    for j in 0..COL {
                        if !self.0[i][j].is_finite() || !other.0[i][j].is_finite() {
                            return false;
                        } else if (self.0[i][j] - other.0[i][j]).abs() > Float::epsilon() {
                            return false;
                        }
                    }
                }
                true
            }
        }
    };
}

impl_partialeq_integral!(i8);
impl_partialeq_integral!(i32);
impl_partialeq_integral!(i64);
impl_partialeq_integral!(i128);
impl_partialeq_integral!(u8);
impl_partialeq_integral!(u32);
impl_partialeq_integral!(u64);
impl_partialeq_integral!(u128);

impl_partialeq_float!(f32);
impl_partialeq_float!(f64);

impl<const ROW: usize, const COL: usize> Eq for Matrix<i8, ROW, COL> {}
impl<const ROW: usize, const COL: usize> Eq for Matrix<i32, ROW, COL> {}
impl<const ROW: usize, const COL: usize> Eq for Matrix<i64, ROW, COL> {}

#[cfg(test)]
pub(crate) fn assert_matrix_approx_eq_float<
    T: PartialEq + NumAssign + Copy + Default + Float + Debug,
    const ROW: usize,
    const COL: usize,
>(
    a: &Matrix<T, ROW, COL>,
    b: &Matrix<T, ROW, COL>,
    tol: T,
) {
    assert_eq!(a.cols(), b.cols());
    assert_eq!(a.rows(), b.rows());
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            if a.0[i][j].is_finite() && !b.0[i][j].is_finite() {
                assert!(false, "matrices not equal:\n{:#?}\n{:#?}", a, b);
            } else if !a.0[i][j].is_finite() && b.0[i][j].is_finite() {
                assert!(false, "matrices not equal:\n{:#?}\n{:#?}", a, b);
            } else if (a.0[i][j] - b.0[i][j]).abs() > tol {
                assert!(false, "matrices not equal:\n{:#?}\n{:#?}", a, b);
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn matrix_approx_eq_float<
    T: PartialEq + NumAssign + Copy + Default + Float + Debug,
    const ROW: usize,
    const COL: usize,
>(
    a: &Matrix<T, ROW, COL>,
    b: &Matrix<T, ROW, COL>,
    tol: T,
) -> bool {
    assert_eq!(a.cols(), b.cols());
    assert_eq!(a.rows(), b.rows());
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            if a.0[i][j].is_finite() && !b.0[i][j].is_finite() {
                return false;
            } else if !a.0[i][j].is_finite() && b.0[i][j].is_finite() {
                return false;
            } else if (a.0[i][j] - b.0[i][j]).abs() > tol {
                return false;
            }
        }
    }
    true
}

pub trait NumericAbs<F> {
    fn inner_abs(&self) -> F;
}

impl NumericAbs<i8> for i8 {
    fn inner_abs(&self) -> Self {
        self.abs()
    }
}

impl NumericAbs<i32> for i32 {
    fn inner_abs(&self) -> Self {
        self.abs()
    }
}
impl NumericAbs<i64> for i64 {
    fn inner_abs(&self) -> Self {
        self.abs()
    }
}
impl NumericAbs<i128> for i128 {
    fn inner_abs(&self) -> Self {
        self.abs()
    }
}
impl NumericAbs<u8> for u8 {
    fn inner_abs(&self) -> Self {
        *self
    }
}
impl NumericAbs<u32> for u32 {
    fn inner_abs(&self) -> Self {
        *self
    }
}
impl NumericAbs<u64> for u64 {
    fn inner_abs(&self) -> Self {
        *self
    }
}
impl NumericAbs<u128> for u128 {
    fn inner_abs(&self) -> Self {
        *self
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> Matrix<T, ROW, COL> {
    pub fn rows(&self) -> usize {
        ROW
    }
    pub fn cols(&self) -> usize {
        COL
    }
    pub fn zero() -> Self {
        let mut m = Self::default();
        for i in 0..ROW {
            for j in 0..COL {
                m[[i, j]] = T::zero();
            }
        }
        m
    }
    pub fn t(&self) -> Matrix<T, COL, ROW> {
        let mut res = Matrix([[T::default(); ROW]; COL]);
        for i in 0..ROW {
            for j in 0..COL {
                res[[j, i]] = self[[i, j]];
            }
        }
        res
    }
    pub fn inner(&self, other: &Self) -> T {
        let mut ret = T::zero();
        for i in 0..ROW {
            for j in 0..COL {
                ret += self[[i, j]] * other[[i, j]];
            }
        }
        ret
    }

    pub fn norm_l1(&self) -> T
    where
        T: std::cmp::PartialOrd + NumericAbs<T>,
    {
        let mut column_l1 = [T::default(); COL];
        for j in 0..COL {
            let mut col_norm = T::zero();
            for i in 0..ROW {
                col_norm += self[[i, j]].inner_abs();
            }
            column_l1[j] = col_norm;
        }
        let mut ret = *column_l1.iter().next().unwrap();
        for &i in column_l1.iter().skip(1) {
            if ret < i {
                ret = i;
            }
        }
        ret
    }

    /// for a vector only, takes 1st column
    pub fn norm_l2(&self) -> T
    where
        T: Real,
    {
        let mut ret = T::zero();
        for j in 0..ROW {
            ret += self[[j, 0]] * self[[j, 0]];
        }
        ret.sqrt()
    }

    pub fn normalize_l2(&self) -> Self
    where
        T: Real,
    {
        let a = self.norm_l2();
        *self / a
    }
}

impl<T: NumAssign + Copy + Default, const ROW: usize> Matrix<T, ROW, ROW> {
    pub fn trace(&self) -> T {
        let mut ret = T::zero();
        for i in 0..ROW {
            ret += self[[i, i]];
        }
        ret
    }
    pub fn eye() -> Self {
        let mut m = Self::zero();
        for i in 0..ROW {
            m[[i, i]] = T::one();
        }
        m
    }
    fn inv_3(m: &Matrix<T, 3, 3>) -> Matrix<T, 3, 3>
    where
        T: Real,
    {
        let determinant = m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[2, 1]] * m[[1, 2]])
            - m[[1, 0]] * (m[[0, 1]] * m[[2, 2]] - m[[2, 1]] * m[[0, 2]])
            + m[[2, 0]] * (m[[0, 1]] * m[[1, 2]] - m[[1, 1]] * m[[0, 2]]);
        // if determinant.abs() < Float::epsilon() {
        //     panic!("det too small");
        // }
        let mut out = Matrix::<T, 3, 3>::default();
        out[[0, 0]] = m[[1, 1]] * m[[2, 2]] - m[[2, 1]] * m[[1, 2]];
        out[[1, 0]] = (m[[0, 1]] * m[[2, 2]] - m[[2, 1]] * m[[0, 2]]).neg();
        out[[2, 0]] = m[[0, 1]] * m[[1, 2]] - m[[1, 1]] * m[[0, 2]];
        out[[0, 1]] = (m[[1, 0]] * m[[2, 2]] - m[[2, 0]] * m[[1, 2]]).neg();
        out[[1, 1]] = m[[0, 0]] * m[[2, 2]] - m[[2, 0]] * m[[0, 2]];
        out[[2, 1]] = (m[[0, 0]] * m[[1, 2]] - m[[1, 0]] * m[[0, 2]]).neg();
        out[[0, 2]] = m[[1, 0]] * m[[2, 1]] - m[[2, 0]] * m[[1, 1]];
        out[[1, 2]] = (m[[0, 0]] * m[[2, 1]] - m[[2, 0]] * m[[0, 1]]).neg();
        out[[2, 2]] = m[[0, 0]] * m[[1, 1]] - m[[1, 0]] * m[[0, 1]];

        (out / determinant).t()
    }

    fn inv_4(m: &Matrix<T, 4, 4>) -> Matrix<T, 4, 4>
    where
        T: Real,
    {
        let mut inv = Matrix::<T, 4, 4>::default();

        inv[[0, 0]] = m[[1, 1]] * m[[2, 2]] * m[[3, 3]]
            - m[[1, 1]] * m[[2, 3]] * m[[3, 2]]
            - m[[2, 1]] * m[[1, 2]] * m[[3, 3]]
            + m[[2, 1]] * m[[1, 3]] * m[[3, 2]]
            + m[[3, 1]] * m[[1, 2]] * m[[2, 3]]
            - m[[3, 1]] * m[[1, 3]] * m[[2, 2]];

        inv[[1, 0]] = -m[[1, 0]] * m[[2, 2]] * m[[3, 3]]
            + m[[1, 0]] * m[[2, 3]] * m[[3, 2]]
            + m[[2, 0]] * m[[1, 2]] * m[[3, 3]]
            - m[[2, 0]] * m[[1, 3]] * m[[3, 2]]
            - m[[3, 0]] * m[[1, 2]] * m[[2, 3]]
            + m[[3, 0]] * m[[1, 3]] * m[[2, 2]];

        inv[[2, 0]] = m[[1, 0]] * m[[2, 1]] * m[[3, 3]]
            - m[[1, 0]] * m[[2, 3]] * m[[3, 1]]
            - m[[2, 0]] * m[[1, 1]] * m[[3, 3]]
            + m[[2, 0]] * m[[1, 3]] * m[[3, 1]]
            + m[[3, 0]] * m[[1, 1]] * m[[2, 3]]
            - m[[3, 0]] * m[[1, 3]] * m[[2, 1]];

        inv[[3, 0]] = -m[[1, 0]] * m[[2, 1]] * m[[3, 2]]
            + m[[1, 0]] * m[[2, 2]] * m[[3, 1]]
            + m[[2, 0]] * m[[1, 1]] * m[[3, 2]]
            - m[[2, 0]] * m[[1, 2]] * m[[3, 1]]
            - m[[3, 0]] * m[[1, 1]] * m[[2, 2]]
            + m[[3, 0]] * m[[1, 2]] * m[[2, 1]];

        inv[[0, 1]] = -m[[0, 1]] * m[[2, 2]] * m[[3, 3]]
            + m[[0, 1]] * m[[2, 3]] * m[[3, 2]]
            + m[[2, 1]] * m[[0, 2]] * m[[3, 3]]
            - m[[2, 1]] * m[[0, 3]] * m[[3, 2]]
            - m[[3, 1]] * m[[0, 2]] * m[[2, 3]]
            + m[[3, 1]] * m[[0, 3]] * m[[2, 2]];

        inv[[1, 1]] = m[[0, 0]] * m[[2, 2]] * m[[3, 3]]
            - m[[0, 0]] * m[[2, 3]] * m[[3, 2]]
            - m[[2, 0]] * m[[0, 2]] * m[[3, 3]]
            + m[[2, 0]] * m[[0, 3]] * m[[3, 2]]
            + m[[3, 0]] * m[[0, 2]] * m[[2, 3]]
            - m[[3, 0]] * m[[0, 3]] * m[[2, 2]];

        inv[[2, 1]] = -m[[0, 0]] * m[[2, 1]] * m[[3, 3]]
            + m[[0, 0]] * m[[2, 3]] * m[[3, 1]]
            + m[[2, 0]] * m[[0, 1]] * m[[3, 3]]
            - m[[2, 0]] * m[[0, 3]] * m[[3, 1]]
            - m[[3, 0]] * m[[0, 1]] * m[[2, 3]]
            + m[[3, 0]] * m[[0, 3]] * m[[2, 1]];

        inv[[3, 1]] = m[[0, 0]] * m[[2, 1]] * m[[3, 2]]
            - m[[0, 0]] * m[[2, 2]] * m[[3, 1]]
            - m[[2, 0]] * m[[0, 1]] * m[[3, 2]]
            + m[[2, 0]] * m[[0, 2]] * m[[3, 1]]
            + m[[3, 0]] * m[[0, 1]] * m[[2, 2]]
            - m[[3, 0]] * m[[0, 2]] * m[[2, 1]];

        inv[[0, 2]] = m[[0, 1]] * m[[1, 2]] * m[[3, 3]]
            - m[[0, 1]] * m[[1, 3]] * m[[3, 2]]
            - m[[1, 1]] * m[[0, 2]] * m[[3, 3]]
            + m[[1, 1]] * m[[0, 3]] * m[[3, 2]]
            + m[[3, 1]] * m[[0, 2]] * m[[1, 3]]
            - m[[3, 1]] * m[[0, 3]] * m[[1, 2]];

        inv[[1, 2]] = -m[[0, 0]] * m[[1, 2]] * m[[3, 3]]
            + m[[0, 0]] * m[[1, 3]] * m[[3, 2]]
            + m[[1, 0]] * m[[0, 2]] * m[[3, 3]]
            - m[[1, 0]] * m[[0, 3]] * m[[3, 2]]
            - m[[3, 0]] * m[[0, 2]] * m[[1, 3]]
            + m[[3, 0]] * m[[0, 3]] * m[[1, 2]];

        inv[[2, 2]] = m[[0, 0]] * m[[1, 1]] * m[[3, 3]]
            - m[[0, 0]] * m[[1, 3]] * m[[3, 1]]
            - m[[1, 0]] * m[[0, 1]] * m[[3, 3]]
            + m[[1, 0]] * m[[0, 3]] * m[[3, 1]]
            + m[[3, 0]] * m[[0, 1]] * m[[1, 3]]
            - m[[3, 0]] * m[[0, 3]] * m[[1, 1]];

        inv[[3, 2]] = -m[[0, 0]] * m[[1, 1]] * m[[3, 2]]
            + m[[0, 0]] * m[[1, 2]] * m[[3, 1]]
            + m[[1, 0]] * m[[0, 1]] * m[[3, 2]]
            - m[[1, 0]] * m[[0, 2]] * m[[3, 1]]
            - m[[3, 0]] * m[[0, 1]] * m[[1, 2]]
            + m[[3, 0]] * m[[0, 2]] * m[[1, 1]];

        inv[[0, 3]] = -m[[0, 1]] * m[[1, 2]] * m[[2, 3]]
            + m[[0, 1]] * m[[1, 3]] * m[[2, 2]]
            + m[[1, 1]] * m[[0, 2]] * m[[2, 3]]
            - m[[1, 1]] * m[[0, 3]] * m[[2, 2]]
            - m[[2, 1]] * m[[0, 2]] * m[[1, 3]]
            + m[[2, 1]] * m[[0, 3]] * m[[1, 2]];

        inv[[1, 3]] = m[[0, 0]] * m[[1, 2]] * m[[2, 3]]
            - m[[0, 0]] * m[[1, 3]] * m[[2, 2]]
            - m[[1, 0]] * m[[0, 2]] * m[[2, 3]]
            + m[[1, 0]] * m[[0, 3]] * m[[2, 2]]
            + m[[2, 0]] * m[[0, 2]] * m[[1, 3]]
            - m[[2, 0]] * m[[0, 3]] * m[[1, 2]];

        inv[[2, 3]] = -m[[0, 0]] * m[[1, 1]] * m[[2, 3]]
            + m[[0, 0]] * m[[1, 3]] * m[[2, 1]]
            + m[[1, 0]] * m[[0, 1]] * m[[2, 3]]
            - m[[1, 0]] * m[[0, 3]] * m[[2, 1]]
            - m[[2, 0]] * m[[0, 1]] * m[[1, 3]]
            + m[[2, 0]] * m[[0, 3]] * m[[1, 1]];

        inv[[3, 3]] = m[[0, 0]] * m[[1, 1]] * m[[2, 2]]
            - m[[0, 0]] * m[[1, 2]] * m[[2, 1]]
            - m[[1, 0]] * m[[0, 1]] * m[[2, 2]]
            + m[[1, 0]] * m[[0, 2]] * m[[2, 1]]
            + m[[2, 0]] * m[[0, 1]] * m[[1, 2]]
            - m[[2, 0]] * m[[0, 2]] * m[[1, 1]];

        let det = m[[0, 0]] * inv[[0, 0]]
            + m[[0, 1]] * inv[[1, 0]]
            + m[[0, 2]] * inv[[2, 0]]
            + m[[0, 3]] * inv[[3, 0]];

        // if det.abs() < Float::epsilon() {
        //     panic!("det too small");
        // }

        let det_2 = T::one() / det;

        inv * det_2
    }

    /// Inverts a square matrix
    ///
    /// # Notes
    ///
    /// - does not check invertibility
    /// - supports matrix size of up to 4x4
    ///
    /// # Examples
    ///
    /// 2x2 inverse:
    /// ```
    /// use lightmatrix::matrix::Matrix;
    /// let a = Matrix::from([[4f32, 7f32], [2f32, 6f32]]);
    /// let ret = a.inv();
    /// let expect = Matrix::from([[0.6f32, -0.7f32], [-0.2f32, 0.4f32]]);
    /// assert_eq!(ret, expect);
    /// ```
    ///
    /// Invalid shape:
    ///
    /// ```compile_error
    /// let a = Matrix::from([[0f32; 3]; 2]);
    /// let result = a.inv(); // a is not square
    /// ```
    pub fn inv(&self) -> Self
    where
        T: Real,
    {
        if ROW == 1 {
            Self::from([[T::one(); ROW]; ROW]) / *self
        } else if ROW == 2 {
            let mut ret = Self::from([[T::default(); ROW]; ROW]);
            let (a, b, c, d) = (self[[0, 0]], self[[0, 1]], self[[1, 0]], self[[1, 1]]);
            ret[[0, 0]] = d;
            ret[[1, 1]] = a;
            ret[[0, 1]] = b.neg();
            ret[[1, 0]] = c.neg();
            ret * (T::one() / (a * d - b * c))
        } else if ROW == 3 {
            use std::mem;
            let m: &Matrix<T, 3, 3> = unsafe { mem::transmute(self) };
            let b = Self::inv_3(m);
            let ret: &Matrix<T, ROW, ROW> = unsafe { mem::transmute(&b) };
            *ret
        } else if ROW == 4 {
            use std::mem;
            let m: &Matrix<T, 4, 4> = unsafe { mem::transmute(self) };
            let b = Self::inv_4(m);
            let ret: &Matrix<T, ROW, ROW> = unsafe { mem::transmute(&b) };
            *ret
        } else {
            unimplemented!();
        }
    }
}

#[test]
fn test_dim() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32]]);
    assert_eq!(a.rows(), 2);
    assert_eq!(a.cols(), 3);
}

#[test]
fn test_transpose() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let expect = Matrix([[0i32, 3i32, 6i32], [1i32, 4i32, 7i32], [2i32, 5i32, 8i32]]);
    let ret = a.t();
    assert_eq!(ret, expect);
}

#[test]
fn test_inner() {
    let a = Matrix([[0i32, 1i32, 2i32]]);
    let b = Matrix([[1i32, 2i32, 3i32]]);
    let expect = 8i32;
    let ret = a.inner(&b);
    assert_eq!(ret, expect);
}

#[test]
fn test_trace() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let expect = 12i32;
    let ret = a.trace();
    assert_eq!(ret, expect);
}

#[test]
fn test_norm_l1_vec() {
    let a = Matrix([[-5i32], [1i32], [2i32]]);
    let expect = 8i32;
    let ret = a.norm_l1();
    assert_eq!(ret, expect);
}

#[test]
fn test_norm_l1_vec_unsigned() {
    let a = Matrix([[5u32], [1u32], [2u32]]);
    let expect = 8u32;
    let ret = a.norm_l1();
    assert_eq!(ret, expect);
}

#[test]
fn test_norm_l2_vec() {
    let a = Matrix([[5f32], [1f32], [2f32]]);
    let expect = 30f32.sqrt();
    let ret = a.norm_l2();
    assert_eq!(ret, expect);
}

#[test]
fn test_inv_mat1() {
    let a = Matrix([[5f32]]);
    let expect = Matrix::from([[0.2f32]]);
    let ret = a.inv();
    assert_eq!(ret, expect);
}

#[test]
fn test_inv_mat2() {
    let a = Matrix::from([[4f32, 7f32], [2f32, 6f32]]);
    let expect = Matrix::from([[0.6f32, -0.7f32], [-0.2f32, 0.4f32]]);
    let ret = a.inv();
    assert_eq!(ret, expect);
}

#[test]
fn test_inv_mat3() {
    let a = Matrix::from([[1f32, 2f32, 3f32], [5f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let expect = Matrix::from([
        [-0.3f32, 0.5f32, -0.2f32],
        [-1f32, -1f32, 1f32],
        [1.1f32, 0.5f32, -0.6f32],
    ]);
    let ret = a.inv();
    assert_eq!(ret, expect);
}

#[test]
fn test_inv_mat4() {
    let a = Matrix::from([
        [3f32, 3f32, 3f32, 1f32],
        [3f32, 2f32, 3f32, 2f32],
        [4f32, 2f32, 1f32, 0f32],
        [2f32, 3f32, 0f32, 3f32],
    ]);
    let expect = Matrix::from([
        [-0.36f32, 0.24f32, 0.36f32, -0.04f32],
        [0.62f32, -0.58f32, -0.12f32, 0.18f32],
        [0.2f32, 0.2f32, -0.2f32, -0.2f32],
        [-0.38f32, 0.42f32, -0.12f32, 0.18f32],
    ]);
    let ret = a.inv();
    assert_eq!(ret, expect);
}

#[test]
fn test_inv_unsupported_size() {
    use std::panic;
    let a = Matrix::from([[0f32; 5]; 5]);
    let result = panic::catch_unwind(|| a.inv());
    assert!(result.is_err());
}

#[cfg(test)]
impl<
        T: NumAssign + Copy + Default + Debug + 'static + From<i32>,
        const ROW: usize,
        const COL: usize,
    > quickcheck::Arbitrary for Matrix<T, ROW, COL>
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let selection = (-100..100).collect::<Vec<_>>();
        let mut m = Matrix::<T, ROW, COL>::default();
        for i in 0..ROW {
            for j in 0..COL {
                m[[i, j]] = T::from(g.choose(&selection).unwrap().to_owned());
            }
        }
        m
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct InvertibleMatrix<T: NumAssign + Copy + Default + 'static, const ROW: usize>(
    pub Matrix<T, ROW, ROW>,
);

#[cfg(test)]
impl<T: NumAssign + Copy + Default + Float + Debug, const ROW: usize> quickcheck::Arbitrary
    for InvertibleMatrix<T, ROW>
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        //generate invertible matrix using elementary matrices

        let mut m = Matrix::<T, ROW, ROW>::eye();

        //arbitrary number of elementary transforms to chain
        for _ in 0..6 {
            let iden = Matrix::<T, ROW, ROW>::eye();

            let mut elementary_transform = Matrix::<T, ROW, ROW>::eye();

            let selection_row = (0..ROW).collect::<Vec<_>>();
            let src_row = g.choose(&selection_row).unwrap().to_owned();
            let dest_row = g.choose(&selection_row).unwrap().to_owned();

            let selection_multiplier = (-3..5).filter(|&x| x != 0).collect::<Vec<_>>();
            let multiplier: i32 = g.choose(&selection_multiplier).unwrap().to_owned();

            for col in 0..ROW {
                let val = T::from(multiplier).unwrap() * iden[[src_row, col]];
                // println!("{:?}", val);
                if src_row == dest_row {
                    elementary_transform[[dest_row, col]] = val;
                } else {
                    elementary_transform[[dest_row, col]] = iden[[dest_row, col]] + val;
                }
            }

            use crate::operator::Dot;
            m = m.dot(&elementary_transform);
        }
        Self(m)
    }
}

#[cfg(test)]
#[quickcheck]
fn matrix_invert_property_test_f64_2x2(m: InvertibleMatrix<f64, 2>) -> bool {
    use crate::operator::Dot;

    let inverse = m.0.inv();
    let expected_eye = inverse.dot(m.0);
    matrix_approx_eq_float(&expected_eye, &Matrix::<f64, 2, 2>::eye(), 1e-5)
}

#[cfg(test)]
#[quickcheck]
fn matrix_invert_property_test_f64_3x3(m: InvertibleMatrix<f64, 3>) -> bool {
    use crate::operator::Dot;

    let inverse = m.0.inv();
    let expected_eye = inverse.dot(m.0);
    matrix_approx_eq_float(&expected_eye, &Matrix::<f64, 3, 3>::eye(), 1e-5)
}

#[cfg(test)]
#[quickcheck]
fn matrix_invert_property_test_f64_4x4(m: InvertibleMatrix<f64, 4>) -> bool {
    use crate::operator::Dot;

    let inverse = m.0.inv();
    let expected_eye = inverse.dot(m.0);
    matrix_approx_eq_float(&expected_eye, &Matrix::<f64, 4, 4>::eye(), 1e-5)
}

#[cfg(test)]
#[quickcheck]
fn matrix_dot((m1, m2): (Matrix<f64, 15, 3>, Matrix<f64, 3, 7>)) -> bool {
    use crate::operator::Dot;

    let ret = m1.dot(&m2);
    let mut expected = Matrix::<f64, 15, 7>::default();
    for i in 0..15 {
        for j in 0..7 {
            for k in 0..3 {
                expected[[i, j]] += m1[[i, k]] * m2[[k, j]];
            }
        }
    }
    matrix_approx_eq_float(&ret, &expected, 1e-7)
}

#[cfg(test)]
#[quickcheck]
fn matrix_subregion_3x3(m: Matrix<f64, 15, 10>) -> bool {
    let sub_region = [(2..5), (3..6)];
    let ret = Matrix::<f64, 3, 3>::from((m, sub_region));

    let mut expected = Matrix::<f64, 3, 3>::default();
    for (i_out, i) in (2..5).enumerate() {
        for (j_out, j) in (3..6).enumerate() {
            expected[[i_out, j_out]] = m[[i, j]];
        }
    }
    matrix_approx_eq_float(&ret, &expected, 1e-7)
}

#[cfg(test)]
#[quickcheck]
fn matrix_subregion_3x1(m: Matrix<f64, 4, 1>) -> bool {
    let ret = Matrix::<f64, 3, 1>::from((m, [(0..3), (0..1)]));

    let mut expected = Matrix::<f64, 3, 1>::default();
    for (i_out, i) in (0..3).enumerate() {
        for (j_out, j) in (0..1).enumerate() {
            expected[[i_out, j_out]] = m[[i, j]];
        }
    }
    matrix_approx_eq_float(&ret, &expected, 1e-7)
}

#[test]
fn matrix_subregion_unmatched_shape() {
    let m = Matrix::<f64, 4, 1>::default();
    use std::panic;
    let result = panic::catch_unwind(|| Matrix::<f64, 3, 1>::from((m, [(0..4), (0..1)])));
    assert!(result.is_err());
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_assign((m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 15, 10>)) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((2..5), (3..6))));
    let mut dest = MatrixSliceMut::<f64, 15, 10>::from((&mut m2, ((5..8), (7..10))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (5..8).enumerate() {
        for (j_idx, j) in (7..10).enumerate() {
            check &= m2[[i, j]] == m1[[2 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_assign_from_larget_region(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 15, 15>),
) -> bool {
    let src = MatrixSlice::from((&m1, ((2..15), (3..15))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((5..8), (7..10))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (5..8).enumerate() {
        for (j_idx, j) in (7..10).enumerate() {
            check &= m2[[i, j]] == m1[[2 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[test]
fn matrix_slice_assign_dest_shape_too_big() {
    use std::panic;
    let result = panic::catch_unwind(|| {
        let (m1, mut m2) = (
            Matrix::<f64, 15, 10>::default(),
            Matrix::<f64, 15, 15>::default(),
        );
        let src = MatrixSlice::from((&m1, ((2..5), (3..6))));
        let mut dest = MatrixSliceMut::from((&mut m2, ((5..9), (7..10))));
        dest.assign(src)
    });
    assert!(result.is_err());
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_assign_range_full((m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 3, 3>)) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((2..5), (3..6))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((..), (..))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..3).enumerate() {
        for (j_idx, j) in (0..3).enumerate() {
            check &= m2[[i, j]] == m1[[2 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_assign_range_inclusive(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 4, 4>),
) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((2..=5), (3..=6))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((..), (..))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= m2[[i, j]] == m1[[2 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_combo_range_and_range_inclusive(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 4, 4>),
) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((2..6), (3..=6))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((..), (..))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= m2[[i, j]] == m1[[2 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_combo_range_and_range_to(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 4, 4>),
) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((..4), (3..7))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((..), (..))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= m2[[i, j]] == m1[[0 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_combo_range_and_range_from(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 4, 4>),
) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((11..), (3..7))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((..), (..))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= m2[[i, j]] == m1[[11 + i_idx, 3 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_combo_range_from_and_range_to_inclusive(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 4, 4>),
) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((11..), (..=3))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((..), (..))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= m2[[i, j]] == m1[[11 + i_idx, 0 + j_idx]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_combo_range_inclusive_and_range_to_inclusive(
    (m1, mut m2): (Matrix<f64, 15, 10>, Matrix<f64, 4, 4>),
) -> bool {
    let src = MatrixSlice::<f64, 15, 10>::from((&m1, ((11..15), (0..4))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((0..=3), (..=3))));
    dest.assign(src);
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= m2[[i, j]] == m1[[11 + i_idx, 0 + j_idx]];
        }
    }
    check
}
