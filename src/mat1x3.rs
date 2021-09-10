use num_traits::{float::Float, int::PrimInt, NumAssign};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};

use crate::inner::{InnerFloat, InnerInt};

use crate::mat3::{Mat3f, Mat3i};
use crate::mat3x1::{Mat3x1f, Mat3x1i};

create_matrix!(Mat1x3i, InnerInt, 1, 3);

impl<T: PrimInt + NumAssign> PartialEq for Mat1x3i<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}
impl<T: PrimInt + NumAssign> Eq for Mat1x3i<T> {}

impl<T: PrimInt + NumAssign> Mat1x3i<T> {
    pub fn t(&self) -> Mat3x1i<T> {
        Mat3x1i::from(self.0.t())
    }
    pub fn dot(&self, other: &Mat3i<T>) -> Self {
        other.t().dot_vec(&self.t()).t()
    }
    pub fn dot_vec(&self, other: &Mat3x1i<T>) -> T {
        self.inner(&other.t())
    }
    pub fn inner(&self, other: &Self) -> T {
        let mut ret = T::zero();
        for i in 0..3 {
            ret += self[[0, i]] * other[[0, i]];
        }
        ret
    }
}

create_matrix!(Mat1x3f, InnerFloat, 1, 3);

impl<T: Float + NumAssign> PartialEq for Mat1x3f<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<T: Float + NumAssign> Mat1x3f<T> {
    pub fn t(&self) -> Mat3x1f<T> {
        Mat3x1f::from(self.0.t())
    }
    pub fn dot(&self, other: &Mat3f<T>) -> Self {
        other.t().dot_vec(&self.t()).t()
    }
    pub fn dot_vec(&self, other: &Mat3x1f<T>) -> T {
        let mut ret = T::zero();
        for i in 0..3 {
            ret += self[[0, i]] * other[[i, 0]];
        }
        ret
    }
    pub fn inner(&self, other: &Self) -> T {
        let mut ret = T::zero();
        for i in 0..3 {
            ret += self[[0, i]] * other[[0, i]];
        }
        ret
    }
}

#[test]
fn test_mat1x3i_t() {
    let a = Mat1x3i::from([[0i32, 1i32, 2i32]]);
    let expect = Mat3x1i::from([[0i32], [1i32], [2i32]]);
    assert_eq!(a.t(), expect);
}

#[test]
fn test_mat1x3f_t() {
    let a = Mat1x3f::from([[0.5f32, 1.5f32, 2.5f32]]);
    let expect = Mat3x1f::from([[0.5f32], [1.5f32], [2.5f32]]);
    assert_eq!(a.t(), expect);
}

#[test]
fn test_mat1x3i_inner_product() {
    let a = Mat1x3i::from([[0i32, 1i32, 2i32]]);
    let b = Mat1x3i::from([[1i32, 2i32, 3i32]]);
    let expect = 8i32;
    assert_eq!(a.inner(&b), expect);
}

#[test]
fn test_mat1x3f_inner_product() {
    let a = Mat1x3f::from([[0f32, 1f32, 2f32]]);
    let b = Mat1x3f::from([[1f32, 2.5f32, 3.5f32]]);
    let expect = 9.5f32;
    assert_eq!(a.inner(&b), expect);
}

#[test]
fn test_mat1x3i_dot_vec() {
    let a = Mat1x3i::from([[0i32, 1i32, 2i32]]);
    let b = Mat3x1i::from([[1i32], [2i32], [3i32]]);
    let expect = 8i32;
    assert_eq!(a.dot_vec(&b), expect);
}

#[test]
fn test_mat1x3f_dot_vec() {
    let a = Mat1x3f::from([[0f32, 1f32, 2f32]]);
    let b = Mat3x1f::from([[1f32], [2f32], [3.5f32]]);
    let expect = 9f32;
    assert_eq!(a.dot_vec(&b), expect);
}

#[test]
fn test_mat1x3i_dot() {
    let a = Mat1x3i::from([[0i32, 1i32, 2i32]]);
    let b = Mat3i::from([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Mat1x3i::from([[18i32, 21i32, 24i32]]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_mat1x3f_dot() {
    let a = Mat1x3f::from([[0f32, 1f32, 2f32]]);
    let b = Mat3f::from([
        [1f32, 2.5f32, 3.5f32],
        [4f32, 5.5f32, 6.5f32],
        [7f32, 8.5f32, 9.5f32],
    ]);
    let expect = Mat1x3f::from([[18f32, 22.5f32, 25.5f32]]);
    assert_eq!(a.dot(&b), expect);
}
