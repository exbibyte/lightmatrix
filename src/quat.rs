#[allow(unused_imports)]
use std::ops::Div;
#[allow(unused_imports)]
use std::ops::Index;
#[allow(unused_imports)]
use std::ops::IndexMut;

// use crate::matrix::Matrix;
use crate::matrix::*;
use crate::scalar::Scalar;

use std::ops::{Add, Mul, Sub};

use num_traits::{real::Real, NumAssign};

#[derive(Clone, Debug)]
pub struct Quat<T: Real + NumAssign + Default + Clone>(pub Matrix<T, 1, 4>);

macro_rules! impl_quat_partialeq_float {
    ($type:ty) => {
        impl PartialEq<Quat<$type>> for Quat<$type> {
            fn eq(&self, other: &Self) -> bool {
                (&self.0).eq(&other.0)
            }
        }
    };
}

impl_quat_partialeq_float!(f32);
impl_quat_partialeq_float!(f64);

impl<T: Real + NumAssign + Default> Default for Quat<T> {
    fn default() -> Quat<T> {
        Quat(Matrix::from([[T::zero(), T::zero(), T::zero(), T::one()]]))
    }
}

impl<T: Real + Default + NumAssign> Quat<T> {
    pub fn x(&self) -> T {
        self.0[[0, 0]]
    }
    pub fn y(&self) -> T {
        self.0[[0, 1]]
    }
    pub fn z(&self) -> T {
        self.0[[0, 2]]
    }
    pub fn w(&self) -> T {
        self.0[[0, 3]]
    }
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[[0, 0]]
    }
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[[0, 1]]
    }
    pub fn z_mut(&mut self) -> &mut T {
        &mut self.0[[0, 2]]
    }
    pub fn w_mut(&mut self) -> &mut T {
        &mut self.0[[0, 3]]
    }
    pub fn init(x: T, y: T, z: T, w: T) -> Self {
        Quat(Matrix::from([[x, y, z, w]]))
    }
    // #[allow(dead_code)]
    // pub fn reflection_in_plane(&self, p: Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
    //     let quat_p = Quat::init(p[[0, 0]], p[[1, 0]], p[[2, 0]], T::zero());
    //     let temp = self * &quat_p;
    //     let temp2 = &temp * self;
    //     Matrix::from([[temp2.x()], [temp2.y()], [temp2.z()]])
    // }
    // #[allow(dead_code)]
    // pub fn parallel_component_of_plane(&self, p: Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
    //     let quat_p = Quat::init(p[[0, 0]], p[[1, 0]], p[[2, 0]], T::zero());
    //     let temp = self * &quat_p;
    //     let temp2 = &temp * self;
    //     let temp3 = &quat_p + &temp2;
    //     let temp4 = &temp3 * T::from(0.5).unwrap();
    //     Matrix::from([[temp4.x()], [temp4.y()], [temp4.z()]])
    // }
    // #[allow(dead_code)]
    // pub fn orthogonal_component_of_plane(&self, p: Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
    //     let quat_p = Quat::init(p[[0, 0]], p[[1, 0]], p[[2, 0]], T::zero());
    //     let temp = self * &quat_p;
    //     let temp2 = &temp * self;
    //     let temp3 = &quat_p - &temp2;
    //     let temp4 = &temp3 * T::from(0.5).unwrap();
    //     Matrix::from([[temp4.x()], [temp4.y()], [temp4.z()]])
    // }
    pub fn add(&self, other: &Self) -> Self {
        Quat::init(
            self.x() + other.x(),
            self.y() + other.y(),
            self.z() + other.z(),
            self.w() + other.w(),
        )
    }
    pub fn minus(&self, other: &Self) -> Self {
        Quat::init(
            self.x() - other.x(),
            self.y() - other.y(),
            self.z() - other.z(),
            self.w() - other.w(),
        )
    }
    pub fn mul(&self, other: &Self) -> Self {
        Quat::init(
            self.w() * other.x() + self.x() * other.w() + self.y() * other.z()
                - self.z() * other.y(),
            self.w() * other.y() - self.x() * other.z()
                + self.y() * other.w()
                + self.z() * other.x(),
            self.w() * other.z() + self.x() * other.y() - self.y() * other.x()
                + self.z() * other.w(),
            self.w() * other.w()
                - self.x() * other.x()
                - self.y() * other.y()
                - self.z() * other.z(),
        )
    }
    pub fn norm_squared(&self) -> T {
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z() + self.w() * self.w()
    }
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }
    pub fn normalize(&self) -> Self {
        let l = self.norm();
        if l > T::zero() || l < T::zero() {
            Quat::init(self.x() / l, self.y() / l, self.z() / l, self.w() / l)
        } else {
            panic!("quat normalization unsuccessful.");
        }
    }
    pub fn normalized(&mut self) {
        let l = self.norm();
        if l > T::zero() || l < T::zero() {
            *self.x_mut() = self.x() / l;
            *self.y_mut() = self.y() / l;
            *self.z_mut() = self.z() / l;
            *self.w_mut() = self.w() / l;
        } else {
            panic!("quat normalization unsuccessful.");
        }
    }
    pub fn ln(&self) -> Self {
        let l = self.norm();
        let w_ln = self.w().ln();
        //normalize x,y,z vector -> v/||v||
        let vec_length = (self.x() * self.x() + self.y() * self.y() + self.z() * self.z()).sqrt();
        assert!(vec_length != T::zero());
        let vec_x = self.x() / vec_length;
        let vec_y = self.y() / vec_length;
        let vec_z = self.z() / vec_length;
        //scale x,y,z by acos( w/l )
        let s = (w_ln / l).acos();
        Quat::init(vec_x * s, vec_y * s, vec_z * s, w_ln)
    }
    pub fn pow(&self, t: T) -> Self {
        let vec_length = (self.x() * self.x() + self.y() * self.y() + self.z() * self.z()).sqrt();
        assert!(vec_length != T::zero());
        let vec_x = self.x() / vec_length;
        let vec_y = self.y() / vec_length;
        let vec_z = self.z() / vec_length;
        let l = self.norm();
        //original angle
        let alpha = (self.w() / l).acos();
        //new angle
        let beta = t * alpha;
        let coeff = l.powf(t);
        Quat::init(
            coeff * vec_x * beta.sin(),
            coeff * vec_y * beta.sin(),
            coeff * vec_z * beta.sin(),
            coeff * beta.cos(),
        )
    }
    pub fn negate(&self) -> Self {
        Quat::init(-self.x(), -self.y(), -self.z(), -self.w())
    }
    pub fn conjugate(&self) -> Self {
        Quat::init(-self.x(), -self.y(), -self.z(), self.w())
    }
    pub fn scale(&self, s: T) -> Self {
        Quat::init(self.x() * s, self.y() * s, self.z() * s, self.w() * s)
    }
    pub fn scaled(&mut self, s: T) {
        *self.x_mut() = self.x() * s;
        *self.y_mut() = self.y() * s;
        *self.z_mut() = self.z() * s;
        *self.w_mut() = self.w() * s;
    }
    pub fn inverse(&self) -> Self {
        let conj = self.conjugate();
        let norm = conj.norm_squared();
        assert!(norm != T::zero());
        &conj * (T::one() / norm)
    }
    pub fn dot(&self, other: &Self) -> T {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
    pub fn lerp(start: Quat<T>, end: Quat<T>, t: T) -> Self {
        let clamp_upper = if t > T::one() { T::one() } else { t };
        let clamp = if clamp_upper < T::zero() {
            T::zero()
        } else {
            clamp_upper
        };
        Quat::init(
            start.x() * (T::one() - clamp) + end.x() * clamp,
            start.y() * (T::one() - clamp) + end.y() * clamp,
            start.z() * (T::one() - clamp) + end.z() * clamp,
            start.w() * (T::one() - clamp) + end.w() * clamp,
        )
    }
    pub fn slerp(start: Quat<T>, end: Quat<T>, t: T) -> Self {
        let t_clamp_upper = if t > T::one() { T::one() } else { t };
        let t_clamp = if t_clamp_upper < T::zero() {
            T::zero()
        } else {
            t_clamp_upper
        };

        let cos_omega =
            start.w() * end.w() + start.x() * end.x() + start.y() * end.y() + start.z() * end.z();
        let cos_omega_adjust = if cos_omega < T::zero() {
            -cos_omega
        } else {
            cos_omega
        };

        let end_adjust = if cos_omega < T::zero() {
            //inverted
            Quat::init(-end.x(), -end.y(), -end.z(), -end.w())
        } else {
            Quat::init(end.x(), end.y(), end.z(), end.w())
        };

        let (k0, k1) = if cos_omega_adjust > T::one() - T::epsilon() {
            (T::one() - t_clamp, t_clamp)
        } else {
            let sin_omega = (T::one() - cos_omega * cos_omega).sqrt();
            let omega = sin_omega.atan2(cos_omega);
            let inv_sin_omega = T::one() / sin_omega;
            (
                ((T::one() - t_clamp) * omega).sin() * inv_sin_omega,
                (t_clamp * omega).sin() * inv_sin_omega,
            )
        };
        Quat::init(
            start.x() * k0 + end_adjust.x() * k1,
            start.y() * k0 + end_adjust.y() * k1,
            start.z() * k0 + end_adjust.z() * k1,
            start.w() * k0 + end_adjust.w() * k1,
        )
    }
}

impl<T: Real + Default + NumAssign> Add for &Quat<T> {
    type Output = Quat<T>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl<T: Real + Default + NumAssign> Add for Quat<T> {
    type Output = Quat<T>;
    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<'a, T: Real + Default + NumAssign> Mul<&'a Quat<T>> for &'a Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<Quat<T>> for Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<T> for &Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<T> for Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<Scalar<T>> for Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: Scalar<T>) -> Self::Output {
        self.scale(rhs.0)
    }
}

impl<T: Real + Default + NumAssign> Mul<&Scalar<T>> for Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: &Scalar<T>) -> Self::Output {
        self.scale(rhs.0)
    }
}

impl<T: Real + Default + NumAssign> Mul<Scalar<T>> for &Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: Scalar<T>) -> Self::Output {
        self.scale(rhs.0)
    }
}

impl<T: Real + Default + NumAssign> Mul<&Scalar<T>> for &Quat<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: &Scalar<T>) -> Self::Output {
        self.scale(rhs.0)
    }
}

impl<T: Real + NumAssign + Copy + Default> Mul<&Quat<T>> for Scalar<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: &Quat<T>) -> Self::Output {
        rhs * self.0
    }
}

impl<T: Real + NumAssign + Copy + Default> Mul<Quat<T>> for Scalar<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: Quat<T>) -> Self::Output {
        rhs * self.0
    }
}

impl<T: Real + Default + NumAssign> Sub for &Quat<T> {
    type Output = Quat<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.minus(rhs)
    }
}

impl<T: Real + Default + NumAssign> Sub for Quat<T> {
    type Output = Quat<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.minus(&rhs)
    }
}

#[test]
fn convert_float() {
    struct Test<T: Real>(std::marker::PhantomData<T>);
    impl<T: Real> Test<T> {
        pub fn give_me_a_number(&self) -> T {
            T::from(0.55).unwrap()
        }
    }
    let t = Test::<f32>(std::marker::PhantomData);
    let num = t.give_me_a_number();
    assert_eq!(0.55f32, num);

    let t2 = Test::<f64>(std::marker::PhantomData);
    let num2 = t2.give_me_a_number();
    assert_eq!(0.55f64, num2);
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct RandQuat<T: Real + NumAssign + Default>(pub Quat<T>);

#[cfg(test)]
impl<T: Real + Default + NumAssign + 'static> quickcheck::Arbitrary for RandQuat<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let selection = (-100..100).map(|x| T::from(x).unwrap()).collect::<Vec<_>>();

        let q = Quat::init(
            g.choose(&selection).unwrap().to_owned(),
            g.choose(&selection).unwrap().to_owned(),
            g.choose(&selection).unwrap().to_owned(),
            g.choose(&selection).unwrap().to_owned(),
        );

        Self(q)
    }
}

#[cfg(test)]
#[quickcheck]
fn quat_mul_scalar_by_quat(m: RandQuat<f64>) -> bool {
    let q = m.0;
    let q2 = Scalar(5.) * &q;
    matrix_approx_eq_float(
        &q2.0,
        &Matrix::<f64, 1, 4>::from([[
            5. * q.0[[0, 0]],
            5. * q.0[[0, 1]],
            5. * q.0[[0, 2]],
            5. * q.0[[0, 3]],
        ]]),
        1e-5,
    )
}

#[cfg(test)]
#[quickcheck]
fn quat_mul_quat_by_float(m: RandQuat<f64>) -> bool {
    let q = m.0;
    let q2 = &q * 5.;
    matrix_approx_eq_float(
        &q2.0,
        &Matrix::<f64, 1, 4>::from([[
            5. * q.0[[0, 0]],
            5. * q.0[[0, 1]],
            5. * q.0[[0, 2]],
            5. * q.0[[0, 3]],
        ]]),
        1e-5,
    )
}

#[cfg(test)]
#[quickcheck]
fn quat_mul_quat_by_scalar(m: RandQuat<f64>) -> bool {
    let q = m.0;
    let q2 = &q * &Scalar(5f64);
    matrix_approx_eq_float(
        &q2.0,
        &Matrix::<f64, 1, 4>::from([[
            5. * q.0[[0, 0]],
            5. * q.0[[0, 1]],
            5. * q.0[[0, 2]],
            5. * q.0[[0, 3]],
        ]]),
        1e-5,
    )
}

#[cfg(test)]
#[quickcheck]
fn quat_partial_eq_same(m: RandQuat<f64>) -> bool {
    m.0.eq(&m.0)
}

#[cfg(test)]
#[quickcheck]
fn quat_partial_eq_different((mut m1, mut m2): (RandQuat<f64>, RandQuat<f64>)) -> bool {
    m1.0 .0[[0, 0]] = 1.0;
    m2.0 .0[[0, 0]] = 2.0;
    !m1.0.eq(&m2.0)
}
