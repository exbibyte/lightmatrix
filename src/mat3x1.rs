use num_traits::{float::Float, int::PrimInt, NumAssign};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};

use crate::inner::{InnerFloat, InnerInt};

use crate::mat1x3::{Mat1x3f, Mat1x3i};
use crate::mat3::{Mat3f, Mat3i};

create_matrix!(Mat3x1i, InnerInt, 3, 1);

impl<T: PrimInt + NumAssign> PartialEq for Mat3x1i<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}
impl<T: PrimInt + NumAssign> Eq for Mat3x1i<T> {}

impl<T: PrimInt + NumAssign> Mat3x1i<T> {
    pub fn t(&self) -> Mat1x3i<T> {
        Mat1x3i::from(self.0.t())
    }
    pub fn dot_vec(&self, other: &Mat1x3i<T>) -> Mat3i<T> {
        let mut ret = Mat3i::zero();
        for i in 0..3 {
            for j in 0..3 {
                ret[[i, j]] = self[[i, 0]] * other[[0, j]];
            }
        }
        ret
    }
    pub fn inner(&self, other: &Self) -> T {
        let mut ret = T::zero();
        for i in 0..3 {
            ret += self[[i, 0]] * other[[i, 0]];
        }
        ret
    }
}

create_matrix!(Mat3x1f, InnerFloat, 3, 1);

impl<T: Float + NumAssign> PartialEq for Mat3x1f<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<T: Float + NumAssign> Mat3x1f<T> {
    pub fn t(&self) -> Mat1x3f<T> {
        Mat1x3f::from(self.0.t())
    }
    pub fn dot_vec(&self, other: &Mat1x3f<T>) -> Mat3f<T> {
        let mut ret = Mat3f::zero();
        for i in 0..3 {
            for j in 0..3 {
                ret[[i, j]] = self[[i, 0]] * other[[0, j]];
            }
        }
        ret
    }
    pub fn inner(&self, other: &Self) -> T {
        let mut ret = T::zero();
        for i in 0..3 {
            ret += self[[i, 0]] * other[[i, 0]];
        }
        ret
    }
}

#[test]
fn test_mat3x1i_t() {
    let a = Mat3x1i::from([[0i32], [1i32], [2i32]]);
    let expect = Mat1x3i::from([[0i32, 1i32, 2i32]]);
    assert_eq!(a.t(), expect);
}

#[test]
fn test_mat3x1f_t() {
    let a = Mat3x1f::from([[0f32], [1f32], [2.5f32]]);
    let expect = Mat1x3f::from([[0f32, 1f32, 2.5f32]]);
    assert_eq!(a.t(), expect);
}

#[test]
fn test_mat3x1i_cross_product() {
    let a = Mat3x1i::from([[0i32], [1i32], [2i32]]);
    let b = Mat1x3i::from([[1i32, 2i32, 3i32]]);
    let expect = Mat3i::from([[0i32, 0i32, 0i32], [1i32, 2i32, 3i32], [2i32, 4i32, 6i32]]);
    assert_eq!(a.dot_vec(&b), expect);
}

#[test]
fn test_mat3x1f_cross_product() {
    let a = Mat3x1f::from([[0f32], [1f32], [2f32]]);
    let b = Mat1x3f::from([[1f32, 2f32, 3.5f32]]);
    let expect = Mat3f::from([[0f32, 0f32, 0f32], [1f32, 2f32, 3.5f32], [2f32, 4f32, 7f32]]);
    assert_eq!(a.dot_vec(&b), expect);
}

#[test]
fn test_mat3x1i_inner_product() {
    let a = Mat3x1i::from([[0i32], [1i32], [2i32]]);
    let b = Mat3x1i::from([[1i32], [2i32], [3i32]]);
    let expect = 8i32;
    assert_eq!(a.inner(&b), expect);
}

#[test]
fn test_mat3x1f_inner_product() {
    let a = Mat3x1f::from([[0f32], [1f32], [2f32]]);
    let b = Mat3x1f::from([[1f32], [2.5f32], [3.5f32]]);
    let expect = 9.5f32;
    assert_eq!(a.inner(&b), expect);
}
