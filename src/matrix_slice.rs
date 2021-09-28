use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use num_traits::NumAssign;
use paste::paste;

use crate::matrix::*;

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

/// convert a matrix slice to an owned matrix
impl<
        'a,
        T: NumAssign + Copy + Default,
        const ROW_DEST: usize,
        const COL_DEST: usize,
        const ROW: usize,
        const COL: usize,
    > From<MatrixSliceMut<'a, T, ROW, COL>> for Matrix<T, ROW_DEST, COL_DEST>
{
    fn from(m: MatrixSliceMut<'a, T, ROW, COL>) -> Self {
        let mut ret = Self::default();

        let (range_rows, range_cols) = m.get_range();
        let src_range_rows_count = range_rows.count;
        let src_range_cols_count = range_cols.count;
        let src_rows_start = range_rows.start;
        let src_cols_start = range_cols.start;

        assert_eq!(src_range_rows_count, ret.rows());
        assert_eq!(src_range_cols_count, ret.cols());

        for i in 0..src_range_rows_count {
            for j in 0..src_range_cols_count {
                ret[[i, j]] = m.matrix[[src_rows_start + i, src_cols_start + j]];
            }
        }
        ret
    }
}

/// convert a matrix slice to an owned matrix
impl<
        'a,
        T: NumAssign + Copy + Default,
        const ROW: usize,
        const COL: usize,
        const ROW_DEST: usize,
        const COL_DEST: usize,
    > From<MatrixSlice<'a, T, ROW, COL>> for Matrix<T, ROW_DEST, COL_DEST>
{
    fn from(m: MatrixSlice<'a, T, ROW, COL>) -> Self {
        let mut ret = Self::default();

        let (range_rows, range_cols) = m.get_range();
        let src_range_rows_count = range_rows.count;
        let src_range_cols_count = range_cols.count;
        let src_rows_start = range_rows.start;
        let src_cols_start = range_cols.start;

        assert_eq!(src_range_rows_count, ret.rows());
        assert_eq!(src_range_cols_count, ret.cols());

        for i in 0..src_range_rows_count {
            for j in 0..src_range_cols_count {
                ret[[i, j]] = m.matrix[[src_rows_start + i, src_cols_start + j]];
            }
        }
        ret
    }
}

/// # Copy values from one slice to another.
/// # Example
/// ```
/// use lightmatrix::{matrix::*,matrix_slice::*};
/// let (mut m1, mut m2) = (
///     Matrix::<f64, 15, 10>::default(),
///     Matrix::<f64, 15, 15>::default(),
/// );
///
/// //fill matrices with values...
///
/// let src = MatrixSlice::from((&m1, ((2..15), (3..15))));
/// let mut dest = MatrixSliceMut::from((&mut m2, ((5..8), (7..10))));
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

#[cfg(test)]
#[quickcheck]
fn matrix_slice_to_owned_matrix(m1: Matrix<i64, 15, 15>) -> bool {
    let slice = MatrixSlice::from((&m1, ((0..10), (0..10))));
    let owned: Matrix<i64, 10, 10> = Matrix::from(slice);
    let mut check = true;
    for i in 0..10 {
        for j in 0..10 {
            check &= m1[[i, j]] == owned[[i, j]];
        }
    }
    check
}

#[test]
fn matrix_slice_unmatched_shape() {
    let m = Matrix::<i64, 4, 1>::default();
    use std::panic;
    let result = panic::catch_unwind(|| {
        let slice = MatrixSlice::from((&m, ((0..4), (0..1))));
        let _owned: Matrix<i64, 4, 3> = Matrix::from(slice);
    });
    assert!(result.is_err());
}
