use crate::matrix::*;
use crate::operator::Dot;
use crate::quat::Quat;
use crate::scalar::Scalar;
use delegate::delegate;
use num_traits::{float::FloatConst, real::Real, NumAssign};
use std::ops::{Add, Mul, Sub};

/// quaternion for translation
#[derive(Clone, Default, Debug)]
pub struct QuatT<T: Real + NumAssign + Default + Clone + PartialEq>(pub(crate) Quat<T>);

impl<T: Real + NumAssign + Default + Clone + PartialEq> From<Quat<T>> for QuatT<T> {
    fn from(q: Quat<T>) -> Self {
        Self(q)
    }
}

impl<T: Real + Default + NumAssign> Add for &QuatT<T> {
    type Output = QuatT<T>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl<T: Real + Default + NumAssign> Add for QuatT<T> {
    type Output = QuatT<T>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl<'a, T: Real + Default + NumAssign> Mul<&'a QuatT<T>> for &'a QuatT<T> {
    type Output = QuatT<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul for QuatT<T> {
    type Output = QuatT<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<T> for &QuatT<T> {
    type Output = QuatT<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<T> for QuatT<T> {
    type Output = QuatT<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Real + NumAssign + Copy + Default> Mul<&QuatT<T>> for Scalar<T> {
    type Output = QuatT<T>;
    fn mul(self, rhs: &QuatT<T>) -> Self::Output {
        rhs * self.0
    }
}

impl<T: Real + NumAssign + Copy + Default> Mul<QuatT<T>> for Scalar<T> {
    type Output = QuatT<T>;
    fn mul(self, rhs: QuatT<T>) -> Self::Output {
        rhs * self.0
    }
}

impl<T: Real + Default + NumAssign> Sub for &QuatT<T> {
    type Output = QuatT<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.minus(rhs)
    }
}

impl<T: Real + Default + NumAssign> Sub for QuatT<T> {
    type Output = QuatT<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.minus(&rhs)
    }
}

impl<T: Real + Default + NumAssign> QuatT<T> {
    pub fn init(x: T, y: T, z: T, w: T) -> Self {
        Self(Quat::init(x, y, z, w))
    }
    pub fn lerp(start: Quat<T>, end: Quat<T>, t: T) -> QuatT<T> {
        Self(Quat::lerp(start, end, t))
    }
    fn slerp(start: Quat<T>, end: Quat<T>, t: T) -> QuatT<T> {
        Self(Quat::slerp(start, end, t))
    }
    pub fn add(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }
    pub fn minus(&self, other: &Self) -> Self {
        Self(&self.0 - &other.0)
    }
    pub fn mul(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }
    pub fn dot(&self, other: &Self) -> T {
        self.0.dot(&other.0)
    }
    delegate! {
        to self.0 {
            pub fn x(&self)->T;
            pub fn y(&self)->T;
            pub fn z(&self)->T;
            pub fn w(&self)->T;
            pub fn x_mut(&mut self) -> & mut T;
            pub fn y_mut(&mut self) -> & mut T;
            pub fn z_mut(&mut self) -> & mut T;
            pub fn w_mut(&mut self) -> & mut T;
            pub fn norm_squared(&self) -> T;
            pub fn norm(&self) -> T;
            #[into]
            pub fn normalize(&self) -> Self;
            pub fn normalized(&mut self);
            #[into]
            pub fn ln(&self) -> Self;
            #[into]
            pub fn pow(&self, t:T) -> Self;
            #[into]
            pub fn negate(&self) -> Self;
            #[into]
            pub fn conjugate(&self) -> Self;
            #[into]
            pub fn scale(&self, s:T) -> Self;
            pub fn scaled(&mut self, s: T);
            #[into]
            pub fn inverse(&self) -> Self;
        }
    }
    #[allow(dead_code)]
    pub fn init_from_translation(trans: Matrix<T, 3, 1>) -> Self {
        let two = T::from(2.).unwrap();
        Self::init(
            trans[[0, 0]] / two,
            trans[[1, 0]] / two,
            trans[[2, 0]] / two,
            T::zero(),
        )
    }
    /// returns a transformation matrix
    #[allow(dead_code)]
    pub fn to_matrix(&self) -> Matrix<T, 4, 4> {
        //assume current quaternion corresponds to translation
        let two = T::from(2.).unwrap();
        Matrix::from([
            [T::zero(), T::zero(), T::zero(), two * self.x()],
            [T::zero(), T::zero(), T::zero(), two * self.y()],
            [T::zero(), T::zero(), T::zero(), two * self.z()],
            [T::zero(), T::zero(), T::zero(), T::one()],
        ])
    }
}

#[test]
fn test_quatt_convert_translation() {
    let t = Matrix::from([[0.55, 1.0, 3.7]]).t();
    let q = QuatT::init_from_translation(t);
    let m = q.to_matrix();
    let v = Matrix::from([[0., 0., 0., 1.]]).t();
    let position = m.dot(&v);
    let p = Matrix::from([[position[[0, 0]], position[[1, 0]], position[[2, 0]]]]).t();
    assert_eq!(p, t);
}
