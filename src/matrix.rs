use num_traits::{float::Float, real::Real, NumAssign, Signed};

use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

// #[cfg(test)]
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct Matrix<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize>(
    pub [[T; COL]; ROW],
);

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

pub(crate) fn matrix_approx_eq_float<
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
            } else if (a.0[i][j] - b.0[i][j]).abs() > tol {
                assert!(false, "matrices not equal:\n{:#?}\n{:#?}", a, b);
            }
        }
    }
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
    /// let a = Matrix::from([[4f32, 7f32], [2f32, 6f32]]);
    /// let ret = a.inv();
    /// let expect = Matrix::from([[0.6f32, -0.7f32], [-0.2f32, 0.4f32]]);
    /// assert_eq!(ret, expect);
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
