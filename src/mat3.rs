use num_traits::{float::Float, int::PrimInt, NumAssign};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};

use crate::inner::{InnerFloat, InnerInt};

use crate::mat3x1::{Mat3x1f, Mat3x1i};

create_matrix!(Mat3i, InnerInt, 3, 3);

impl<T: PrimInt + NumAssign> PartialEq for Mat3i<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}
impl<T: PrimInt + NumAssign> Eq for Mat3i<T> {}

impl<T: PrimInt + NumAssign> Mat3i<T> {
    pub fn t(&self) -> Self {
        Self(self.0.t())
    }
    pub fn dot(&self, other: &Self) -> Self {
        Self(self.0.dot(&other.0))
    }
    pub fn dot_vec(&self, other: &Mat3x1i<T>) -> Mat3x1i<T> {
        Mat3x1i::from(self.0.dot_vec(&other.0))
    }
}

create_matrix!(Mat3f, InnerFloat, 3, 3);

impl<T: Float + NumAssign> PartialEq for Mat3f<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other.deref())
    }
}

impl<T: Float + NumAssign> Mat3f<T> {
    pub fn t(&self) -> Self {
        Self(self.0.t())
    }
    pub fn dot(&self, other: &Self) -> Self {
        Self(self.0.dot(&other.0))
    }
    pub fn dot_vec(&self, other: &Mat3x1f<T>) -> Mat3x1f<T> {
        Mat3x1f::from(self.0.dot_vec(&other.0))
    }
}

#[test]
fn test_mat3i_op_assign() {
    let a = Mat3i::from([[0i32, 1i32, 0i32], [2i32, 3i32, 0i32], [4i32, 5i32, 0i32]]);
    let mut b = Mat3i::from([[0i32, 2i32, 1i32], [4i32, 6i32, 2i32], [8i32, 10i32, 3i32]]);
    b -= a;
    let expect = Mat3i::from([[0i32, 1i32, 1i32], [2i32, 3i32, 2i32], [4i32, 5i32, 3i32]]);
    assert_eq!(b, expect);
}

#[test]
fn test_mat3i_op() {
    let a = Mat3i::from([[0i32, 1i32, 0i32], [2i32, 3i32, 0i32], [4i32, 5i32, 0i32]]);
    let b = Mat3i::from([[0i32, 2i32, 1i32], [4i32, 6i32, 2i32], [8i32, 10i32, 3i32]]);
    let c = a * b;
    let expect = Mat3i::from([
        [0i32, 2i32, 0i32],
        [8i32, 18i32, 0i32],
        [32i32, 50i32, 0i32],
    ]);
    assert_eq!(c, expect);
}

#[test]
fn test_mat3i_op_assign_scalar() {
    let mut a = Mat3i::from([[0i32, 1i32, 0i32], [2i32, 3i32, 0i32], [4i32, 5i32, 0i32]]);
    let b = 2i32;
    a *= b;
    let expect = Mat3i::from([[0i32, 2i32, 0i32], [4i32, 6i32, 0i32], [8i32, 10i32, 0i32]]);
    assert_eq!(a, expect);
}

#[test]
fn test_mat3i_op_scalar() {
    let a = Mat3i::from([[0i32, 1i32, 0i32], [2i32, 3i32, 0i32], [4i32, 5i32, 0i32]]);
    let b = 3i32;
    let c = a * b;
    let expect = Mat3i::from([[0i32, 3i32, 0i32], [6i32, 9i32, 0i32], [12i32, 15i32, 0i32]]);
    assert_eq!(c, expect);
}

#[test]
fn test_mat3f_op_scalar() {
    let a = Mat3f::from([[0f32, 1f32, 0f32], [2f32, 3f32, 0f32], [4f32, 5.5f32, 0f32]]);
    let b = 3f32;
    let c = a * b;
    let expect = Mat3f::from([
        [0f32, 3f32, 0f32],
        [6f32, 9f32, 0f32],
        [12f32, 16.5f32, 0f32],
    ]);
    assert_eq!(c, expect);
}

#[test]
fn test_mat3i_t() {
    let a = Mat3i::from([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let expect = Mat3i::from([[0i32, 3i32, 6i32], [1i32, 4i32, 7i32], [2i32, 5i32, 8i32]]);
    assert_eq!(a.t(), expect);
}

#[test]
fn test_mat3i_dot() {
    let a = Mat3i::from([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Mat3i::from([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Mat3i::from([
        [18i32, 21i32, 24i32],
        [54i32, 66i32, 78i32],
        [90i32, 111i32, 132i32],
    ]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_mat3i_dot_vec() {
    let a = Mat3i::from([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Mat3x1i::from([[1i32], [4i32], [7i32]]);
    let expect = Mat3x1i::from([[18i32], [54i32], [90i32]]);
    assert_eq!(a.dot_vec(&b), expect);
}

#[test]
fn test_mat3f_op_assign() {
    let a = Mat3f::from([[0f32, 1f32, 0f32], [2f32, 3f32, 0f32], [4f32, 5f32, 0f32]]);
    let mut b = Mat3f::from([[0f32, 2f32, 1f32], [4f32, 6f32, 2f32], [8f32, 10f32, 3f32]]);
    b -= a;
    let expect = Mat3f::from([[0f32, 1f32, 1f32], [2f32, 3f32, 2f32], [4f32, 5f32, 3f32]]);
    assert_eq!(b, expect);
}

#[test]
fn test_mat3f_op() {
    let a = Mat3f::from([[0f32, 1f32, 0f32], [2f32, 3f32, 0f32], [4f32, 5f32, 0f32]]);
    let b = Mat3f::from([[0f32, 2f32, 1f32], [4f32, 6f32, 2f32], [8f32, 10f32, 3f32]]);
    let c = a * b;
    let expect = Mat3f::from([
        [0f32, 2f32, 0f32],
        [8f32, 18f32, 0f32],
        [32f32, 50f32, 0f32],
    ]);
    assert_eq!(c, expect);
}

#[test]
fn test_mat3f_t() {
    let a = Mat3f::from([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let expect = Mat3f::from([[0f32, 3f32, 6f32], [1f32, 4f32, 7f32], [2f32, 5f32, 8f32]]);
    assert_eq!(a.t(), expect);
}

#[test]
fn test_mat3f_dot() {
    let a = Mat3f::from([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = Mat3f::from([[1f32, 2f32, 3f32], [4f32, 5f32, 6f32], [7f32, 8f32, 9f32]]);
    let expect = Mat3f::from([
        [18f32, 21f32, 24f32],
        [54f32, 66f32, 78f32],
        [90f32, 111f32, 132f32],
    ]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_mat3f_dot_vec() {
    let a = Mat3f::from([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = Mat3x1f::from([[1f32], [4f32], [7f32]]);
    let expect = Mat3x1f::from([[18f32], [54f32], [90f32]]);
    assert_eq!(a.dot_vec(&b), expect);
}
