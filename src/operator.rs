use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use num_traits::NumAssign;

use crate::matrix::Matrix;
use crate::matrix_slice::*;
use crate::scalar::Scalar;

pub trait Dot<Rhs> {
    type R;
    fn dot(&self, rhs: Rhs) -> Self::R;
}

// pub fn dot2<Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> <Lhs as Dot<Rhs>>::R
// where
//     Lhs: Dot<Rhs>,
// {
//     lhs.dot2(rhs)
// }

impl<T: NumAssign + Copy + Default, const ROW_1: usize, const COL_1: usize, const COL_2: usize>
    Dot<&Matrix<T, COL_1, COL_2>> for Matrix<T, ROW_1, COL_1>
{
    type R = Matrix<T, ROW_1, COL_2>;
    fn dot(&self, other: &Matrix<T, COL_1, COL_2>) -> Matrix<T, ROW_1, COL_2> {
        let mut ret = Matrix([[T::default(); COL_2]; ROW_1]);
        for i in 0..ROW_1 {
            for j in 0..COL_2 {
                for k in 0..COL_1 {
                    ret[[i, j]] += self[[i, k]] * other[[k, j]];
                }
            }
        }
        ret
    }
}

impl<T: NumAssign + Copy + Default, const ROW_1: usize, const COL_1: usize, const COL_2: usize>
    Dot<Matrix<T, COL_1, COL_2>> for Matrix<T, ROW_1, COL_1>
{
    type R = Matrix<T, ROW_1, COL_2>;
    fn dot(&self, other: Matrix<T, COL_1, COL_2>) -> Matrix<T, ROW_1, COL_2> {
        let mut ret = Matrix([[T::default(); COL_2]; ROW_1]);
        for i in 0..ROW_1 {
            for j in 0..COL_2 {
                for k in 0..COL_1 {
                    ret[[i, j]] += self[[i, k]] * other[[k, j]];
                }
            }
        }
        ret
    }
}

/// work around unsupported const generic associated types
pub fn dot<
    'a,
    T: 'a + NumAssign + Copy + Default,
    const ROW_1: usize,
    const COL_1: usize,
    const ROW_2: usize,
    const COL_2: usize,
    const ROW_RET: usize,
    const COL_RET: usize,
>(
    this: MatrixSlice<'a, T, COL_1, ROW_1>,
    other: MatrixSlice<'a, T, COL_2, COL_2>,
) -> Matrix<T, ROW_RET, COL_RET> {
    let mut ret = Matrix([[T::default(); COL_RET]; ROW_RET]);

    let (
        RangeInfo {
            start: row_start_self,
            count: row_count_self,
        },
        RangeInfo {
            start: col_start_self,
            count: col_count_self,
        },
    ) = this.get_range();

    let (
        RangeInfo {
            start: row_start_other,
            count: row_count_other,
        },
        RangeInfo {
            start: col_start_other,
            count: col_count_other,
        },
    ) = other.get_range();

    assert_eq!(col_count_self, row_count_other);
    assert_eq!(ROW_RET, row_count_self);
    assert_eq!(COL_RET, col_count_other);

    for i in 0..ROW_RET {
        for j in 0..COL_RET {
            for k in 0..col_count_self {
                ret[[i, j]] += this.matrix[[row_start_self + i, col_start_self + k]]
                    * other.matrix[[row_start_other + k, col_start_other + j]];
            }
        }
    }
    ret
}

macro_rules! delegate_num_ops {
    ($trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait
            for Matrix<T, ROW, COL>
        {
            type Output = Self;

            fn $func(self, rhs: Self) -> Self {
                let mut ret = Matrix::default();

                for y in 0..ROW {
                    for x in 0..COL {
                        ret[[y, x]] = self[[y, x]].$func(rhs[[y, x]]);
                    }
                }

                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign {
    ($trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait
            for Matrix<T, ROW, COL>
        {
            fn $func(&mut self, rhs: Self) {
                for y in 0..ROW {
                    for x in 0..COL {
                        self[[y, x]].$func(rhs[[y, x]]);
                    }
                }
            }
        }
    };
}

macro_rules! delegate_num_ops_scalar {
    ( $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait<T>
            for Matrix<T, ROW, COL>
        {
            type Output = Self;

            fn $func(self, rhs: T) -> Self {
                let mut ret = Self::default();

                for y in 0..ROW {
                    for x in 0..COL {
                        ret[[y, x]] = self[[y, x]].$func(rhs);
                    }
                }

                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_scalar {
    ($trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait<T>
            for Matrix<T, ROW, COL>
        {
            fn $func(&mut self, rhs: T) {
                for y in 0..ROW {
                    for x in 0..COL {
                        self[[y, x]].$func(rhs);
                    }
                }
            }
        }
    };
}

macro_rules! delegate_num_ops_scalar_left {
    ($trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>
            $trait<Matrix<T, ROW, COL>> for Scalar<T>
        {
            type Output = Matrix<T, ROW, COL>;

            fn $func(self, rhs: Matrix<T, ROW, COL>) -> Self::Output {
                rhs.$func(self.0)
            }
        }
    };
}

delegate_num_ops!(Mul, mul);
delegate_num_ops!(Div, div);
delegate_num_ops!(Add, add);
delegate_num_ops!(Sub, sub);
delegate_num_ops!(Rem, rem);

delegate_num_ops_assign!(MulAssign, mul_assign);
delegate_num_ops_assign!(DivAssign, div_assign);
delegate_num_ops_assign!(AddAssign, add_assign);
delegate_num_ops_assign!(SubAssign, sub_assign);
delegate_num_ops_assign!(RemAssign, rem_assign);

delegate_num_ops_scalar!(Mul, mul);
delegate_num_ops_scalar!(Div, div);
delegate_num_ops_scalar!(Add, add);
delegate_num_ops_scalar!(Sub, sub);
delegate_num_ops_scalar!(Rem, rem);

delegate_num_ops_scalar_left!(Mul, mul);
delegate_num_ops_scalar_left!(Div, div);
delegate_num_ops_scalar_left!(Add, add);
delegate_num_ops_scalar_left!(Sub, sub);
delegate_num_ops_scalar_left!(Rem, rem);

delegate_num_ops_assign_scalar!(SubAssign, sub_assign);
delegate_num_ops_assign_scalar!(AddAssign, add_assign);
delegate_num_ops_assign_scalar!(MulAssign, mul_assign);
delegate_num_ops_assign_scalar!(DivAssign, div_assign);
delegate_num_ops_assign_scalar!(RemAssign, rem_assign);

macro_rules! delegate_num_ops_assign_matrix_slice {
    ($trait:ident, $func:ident) => {
        impl<
                'a,
                T: 'a + NumAssign + Copy + Default,
                const ROW1: usize,
                const COL1: usize,
                const ROW2: usize,
                const COL2: usize,
            > $trait<MatrixSlice<'a, T, ROW2, COL2>> for MatrixSliceMut<'a, T, ROW1, COL1>
        {
            fn $func(&mut self, rhs: MatrixSlice<'a, T, ROW2, COL2>) {
                let (range_lhs_rows, range_lhs_cols) = self.get_range();
                let (range_rhs_rows, range_rhs_cols) = rhs.get_range();
                assert_eq!(range_lhs_rows.count, range_rhs_rows.count);
                assert_eq!(range_lhs_cols.count, range_rhs_cols.count);

                for i in 0..range_lhs_rows.count {
                    for j in 0..range_lhs_cols.count {
                        self.matrix[[range_lhs_rows.start + i, range_lhs_cols.start + j]].$func(
                            rhs.matrix[[range_rhs_rows.start + i, range_rhs_cols.start + j]],
                        );
                    }
                }
            }
        }
        impl<
                'a,
                T: 'a + NumAssign + Copy + Default,
                const ROW1: usize,
                const COL1: usize,
                const ROW2: usize,
                const COL2: usize,
            > $trait<MatrixSliceMut<'a, T, ROW2, COL2>> for MatrixSliceMut<'a, T, ROW1, COL1>
        {
            fn $func(&mut self, rhs: MatrixSliceMut<'a, T, ROW2, COL2>) {
                let (range_lhs_rows, range_lhs_cols) = self.get_range();
                let (range_rhs_rows, range_rhs_cols) = rhs.get_range();
                assert_eq!(range_lhs_rows.count, range_rhs_rows.count);
                assert_eq!(range_lhs_cols.count, range_rhs_cols.count);

                for i in 0..range_lhs_rows.count {
                    for j in 0..range_lhs_cols.count {
                        self.matrix[[range_lhs_rows.start + i, range_lhs_cols.start + j]].$func(
                            rhs.matrix[[range_rhs_rows.start + i, range_rhs_cols.start + j]],
                        );
                    }
                }
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_scalar_matrix_slice {
    ($trait:ident, $func:ident) => {
        impl<'a, T: 'a + NumAssign + Copy + Default, const ROW1: usize, const COL1: usize> $trait<T>
            for MatrixSliceMut<'a, T, ROW1, COL1>
        {
            fn $func(&mut self, rhs: T) {
                let (range_lhs_rows, range_lhs_cols) = self.get_range();

                for i in 0..range_lhs_rows.count {
                    for j in 0..range_lhs_cols.count {
                        self.matrix[[range_lhs_rows.start + i, range_lhs_cols.start + j]]
                            .$func(rhs);
                    }
                }
            }
        }
    };
}

delegate_num_ops_assign_matrix_slice!(MulAssign, mul_assign);
delegate_num_ops_assign_matrix_slice!(DivAssign, div_assign);
delegate_num_ops_assign_matrix_slice!(AddAssign, add_assign);
delegate_num_ops_assign_matrix_slice!(SubAssign, sub_assign);
delegate_num_ops_assign_matrix_slice!(RemAssign, rem_assign);

delegate_num_ops_assign_scalar_matrix_slice!(MulAssign, mul_assign);
delegate_num_ops_assign_scalar_matrix_slice!(DivAssign, div_assign);
delegate_num_ops_assign_scalar_matrix_slice!(AddAssign, add_assign);
delegate_num_ops_assign_scalar_matrix_slice!(SubAssign, sub_assign);
delegate_num_ops_assign_scalar_matrix_slice!(RemAssign, rem_assign);

#[cfg(test)]
#[quickcheck]
fn matrix_slice_ops_assign((m1, mut m2): (Matrix<i64, 15, 10>, Matrix<i64, 4, 4>)) -> bool {
    let dest_copy = m2.clone();
    let src = MatrixSlice::from((&m1, ((11..15), (0..4))));
    let mut dest = MatrixSliceMut::from((&mut m2, ((0..=3), (..=3))));
    dest *= src;
    let mut check = true;
    for (i_idx, i) in (0..4).enumerate() {
        for (j_idx, j) in (0..4).enumerate() {
            check &= dest_copy[[i, j]] * m1[[11 + i_idx, 0 + j_idx]] == m2[[i, j]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_ops_assign_self(mut m: Matrix<i64, 4, 4>) -> bool {
    let dest_copy = m.clone();
    let mut src = MatrixSliceMut::from((&mut m, ((..), (..))));
    src *= MatrixSlice::<i64, 4, 4>::from((&Matrix::from(&src), ((..), (..))));
    let mut check = true;
    for i in 0..4 {
        for j in 0..4 {
            check &= dest_copy[[i, j]] * dest_copy[[i, j]] == m[[i, j]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_ops_assign_mut((mut m1, mut m2): (Matrix<i64, 4, 4>, Matrix<i64, 4, 4>)) -> bool {
    let m1_copy = m1.clone();
    let m2_copy = m2.clone();
    let mut slice_1 = MatrixSliceMut::from((&mut m1, ((..), (..))));
    let slice_2 = MatrixSliceMut::from((&mut m2, ((..), (..))));
    slice_1 *= slice_2;
    let mut check = true;
    for i in 0..4 {
        for j in 0..4 {
            check &= m1_copy[[i, j]] * m2_copy[[i, j]] == m1[[i, j]];
        }
    }
    check
}

#[cfg(test)]
#[quickcheck]
fn matrix_slice_ops_assign_scalar(mut m: Matrix<i64, 4, 4>) -> bool {
    let m_copy = m.clone();
    let mut slice = MatrixSliceMut::from((&mut m, ((..), (..))));
    slice *= 3;
    let mut check = true;
    for i in 0..4 {
        for j in 0..4 {
            check &= m_copy[[i, j]] * 3 == m[[i, j]];
        }
    }
    check
}

impl<T: Neg<Output = T> + NumAssign + Default + Copy, const ROW: usize, const COL: usize> Neg
    for Matrix<T, ROW, COL>
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut m = Self::default();
        for i in 0..ROW {
            for j in 0..COL {
                m[[i, j]] = -self[[i, j]];
            }
        }
        m
    }
}

#[test]
fn test_rawarray_operator_dot() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Matrix([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Matrix([
        [18i32, 21i32, 24i32],
        [54i32, 66i32, 78i32],
        [90i32, 111i32, 132i32],
    ]);
    let ret = a.dot(&b);
    assert_eq!(ret, expect);
}

#[test]
fn test_rawarray_operator_dot_vec() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Matrix([[1i32], [4i32], [7i32]]);
    let expect = Matrix([[18i32], [54i32], [90i32]]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_rawarray_operator_dot_vec_float() {
    let a = Matrix([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = Matrix([[1f32], [4f32], [7f32]]);
    let expect = Matrix([[18f32], [54f32], [90f32]]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_rawarray_operator_mul() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Matrix([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Matrix([
        [0i32, 2i32, 6i32],
        [12i32, 20i32, 30i32],
        [42i32, 56i32, 72i32],
    ]);
    let ret = a * b;
    assert_eq!(ret, expect);
}

#[test]
fn test_rawarray_operator_mul_scalar() {
    let a = Matrix([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = 6f32;
    let expect = Matrix([
        [0f32, 6f32, 12f32],
        [18f32, 24f32, 30f32],
        [36f32, 42f32, 48f32],
    ]);
    let ret = a * b;
    assert_eq!(ret, expect);
}

#[test]
fn test_rawarray_operator_mul_scalar_left() {
    let a = Matrix([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = 6f32;
    let expect = Matrix([
        [0f32, 6f32, 12f32],
        [18f32, 24f32, 30f32],
        [36f32, 42f32, 48f32],
    ]);
    let ret = Scalar::from(b) * a;
    assert_eq!(ret, expect);
}

#[cfg(test)]
#[quickcheck]
fn matrix_negate(m: Matrix<f64, 4, 3>) -> bool {
    use crate::matrix::matrix_approx_eq_float;

    let ret = -m;
    let mut expected = m.clone();
    for i in 0..4 {
        for j in 0..3 {
            expected[[i, j]] = -expected[[i, j]];
        }
    }
    matrix_approx_eq_float(&ret, &expected, 1e-7)
}

#[cfg(test)]
use paste::paste;

#[cfg(test)]
macro_rules! matrix_elem_op_check {
    ($func:ident) => {
        paste! {
            mod [<matrix_elem_ $func>] {
                #[quickcheck]
                fn test((m1, m2): (crate::matrix::Matrix<f64, 4, 3>, crate::matrix::Matrix<f64, 4, 3>)) -> bool {
                    use crate::matrix::matrix_approx_eq_float;

                    use std::ops::*;
                    let ret = m1.$func(m2);
                    let mut expected = m1.clone();
                    for i in 0..4 {
                        for j in 0..3 {
                            expected[[i, j]] = m1[[i, j]].$func(m2[[i, j]]);
                        }
                    }
                    matrix_approx_eq_float(&ret, &expected, 1e-7)
                }
            }
        }
    };
}

#[cfg(test)]
matrix_elem_op_check!(add);
#[cfg(test)]
matrix_elem_op_check!(sub);
#[cfg(test)]
matrix_elem_op_check!(mul);
#[cfg(test)]
matrix_elem_op_check!(div);
#[cfg(test)]
matrix_elem_op_check!(rem);
