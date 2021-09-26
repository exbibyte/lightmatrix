use crate::matrix::*;
#[cfg(test)]
use crate::operator::Dot;
use crate::quat::Quat;
use crate::scalar::Scalar;
use delegate::delegate;
use num_traits::{float::FloatConst, real::Real, NumAssign};
use std::ops::{Add, Mul, Sub};

/// quaternion for rotation
#[derive(Clone, Default, Debug)]
pub struct QuatR<T: Real + NumAssign + Default + Clone + PartialEq>(pub(crate) Quat<T>);

macro_rules! impl_quatr_partialeq_float {
    ($type:ty) => {
        impl PartialEq<QuatR<$type>> for QuatR<$type> {
            fn eq(&self, other: &Self) -> bool {
                (&self.0).eq(&other.0)
            }
        }
    };
}

impl_quatr_partialeq_float!(f32);
impl_quatr_partialeq_float!(f64);

impl<T: Real + NumAssign + Default + Clone + PartialEq> From<Quat<T>> for QuatR<T> {
    fn from(q: Quat<T>) -> Self {
        Self(q)
    }
}

impl<T: Real + Default + NumAssign> Add for &QuatR<T> {
    type Output = QuatR<T>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl<T: Real + Default + NumAssign> Add for QuatR<T> {
    type Output = QuatR<T>;
    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<'a, T: Real + Default + NumAssign> Mul<&'a QuatR<T>> for &'a QuatR<T> {
    type Output = QuatR<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<QuatR<T>> for QuatR<T> {
    type Output = QuatR<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<T> for &QuatR<T> {
    type Output = QuatR<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Real + Default + NumAssign> Mul<T> for QuatR<T> {
    type Output = QuatR<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: Real + NumAssign + Copy + Default> Mul<&QuatR<T>> for Scalar<T> {
    type Output = QuatR<T>;
    fn mul(self, rhs: &QuatR<T>) -> Self::Output {
        rhs * self.0
    }
}

impl<T: Real + NumAssign + Copy + Default> Mul<QuatR<T>> for Scalar<T> {
    type Output = QuatR<T>;
    fn mul(self, rhs: QuatR<T>) -> Self::Output {
        rhs * self.0
    }
}

impl<T: Real + Default + NumAssign> Sub for &QuatR<T> {
    type Output = QuatR<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.minus(rhs)
    }
}

impl<T: Real + Default + NumAssign> Sub for QuatR<T> {
    type Output = QuatR<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        (&self).minus(&rhs)
    }
}

impl<T: Real + Default + NumAssign> QuatR<T> {
    pub fn init(x: T, y: T, z: T, w: T) -> Self {
        Self(Quat::init(x, y, z, w))
    }
    pub fn lerp(start: Quat<T>, end: Quat<T>, t: T) -> QuatR<T> {
        Self(Quat::lerp(start, end, t))
    }
    pub fn slerp(start: Quat<T>, end: Quat<T>, t: T) -> QuatR<T> {
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
    pub fn init_auto_w(x: T, y: T, z: T) -> Self {
        let w = T::one() - x * x - y * y - z * z;
        if w < T::zero() {
            QuatR(Quat(Matrix::from([[x, y, z, w]])))
        } else {
            QuatR(Quat(Matrix::from([[
                x,
                y,
                z,
                T::from(-1.).unwrap() * w.sqrt(),
            ]])))
        }
    }
    ///expects a proper rotation matrix as input
    pub fn init_from_rotation(rot: Matrix<T, 3, 3>) -> Self {
        let t = rot.trace();
        if t > T::zero() {
            let s = T::from(0.5).unwrap() / (t + T::one()).sqrt();

            QuatR(Quat::init(
                (rot[[2, 1]] - rot[[1, 2]]) * s,
                (rot[[0, 2]] - rot[[2, 0]]) * s,
                (rot[[1, 0]] - rot[[0, 1]]) * s,
                T::one() / s * T::from(0.25).unwrap(),
            ))
        } else if rot[[0, 0]] > rot[[1, 1]] && rot[[0, 0]] > rot[[2, 2]] {
            let s =
                T::from(2.).unwrap() * (T::one() + rot[[0, 0]] - rot[[1, 1]] - rot[[2, 2]]).sqrt();

            QuatR(Quat::init(
                T::from(0.25).unwrap() * s,
                (rot[[0, 1]] + rot[[1, 0]]) / s,
                (rot[[0, 2]] + rot[[2, 0]]) / s,
                (rot[[2, 1]] - rot[[1, 2]]) / s,
            ))
        } else if rot[[1, 1]] > rot[[2, 2]] {
            let s =
                T::from(2.).unwrap() * (T::one() + rot[[1, 1]] - rot[[0, 0]] - rot[[2, 2]]).sqrt();

            QuatR(Quat::init(
                (rot[[0, 1]] - rot[[1, 0]]) / s,
                T::from(0.25).unwrap() * s,
                (rot[[1, 2]] - rot[[2, 1]]) / s,
                (rot[[0, 2]] - rot[[2, 0]]) / s,
            ))
        } else {
            let s =
                T::from(2.).unwrap() * (T::one() + rot[[2, 2]] - rot[[0, 0]] - rot[[1, 1]]).sqrt();

            QuatR(Quat::init(
                (rot[[0, 2]] - rot[[2, 0]]) / s,
                (rot[[1, 2]] - rot[[2, 1]]) / s,
                T::from(0.25).unwrap() * s,
                (rot[[1, 0]] - rot[[0, 1]]) / s,
            ))
        }
    }
    /// returns a transformation matrix
    #[allow(dead_code)]
    pub fn to_matrix(&self) -> Matrix<T, 4, 4> {
        //assumes unit quaternion
        let a = self.normalize();
        let two = T::from(2.).unwrap();

        Matrix::from([
            [
                T::one() - two * (a.y() * a.y() + a.z() * a.z()), //first row
                two * (a.x() * a.y() - a.z() * a.w()),
                two * (a.x() * a.z() + a.y() * a.w()),
                T::zero(),
            ],
            [
                two * (a.x() * a.y() + a.z() * a.w()), //second row
                T::one() - two * (a.x() * a.x() + a.z() * a.z()),
                two * (a.y() * a.z() - a.x() * a.w()),
                T::zero(),
            ],
            [
                two * (a.x() * a.z() - a.y() * a.w()), //third row
                two * (a.z() * a.y() + a.x() * a.w()),
                T::one() - two * (a.x() * a.x() + a.y() * a.y()),
                T::zero(),
            ],
            [
                T::zero(), //last row
                T::zero(),
                T::zero(),
                T::one(),
            ],
        ])
    }
    #[allow(dead_code)]
    pub fn init_from_axis_angle_degree(axis: Matrix<T, 3, 1>, angle: T) -> Self
    where
        T: FloatConst,
    {
        Self::init_from_axis_angle_radian(axis, angle.to_radians())
    }
    #[allow(dead_code)]
    pub fn init_from_axis_angle_radian(axis: Matrix<T, 3, 1>, angle: T) -> Self
    where
        T: FloatConst,
    {
        let two = T::from(2.).unwrap();
        let radian = ((angle % (two * T::PI())) + two * T::PI()) % (two * T::PI());
        let axis_adjust = axis.normalize_l2();
        let sine_half = (radian / T::from(2.).unwrap()).sin();
        QuatR(Quat::init(
            axis_adjust[[0, 0]] * sine_half,
            axis_adjust[[1, 0]] * sine_half,
            axis_adjust[[2, 0]] * sine_half,
            (radian / two).cos(),
        ))
    }
    /// returns ([x,y,z], angle) where angle is in radians
    #[allow(dead_code)]
    pub fn to_axis_angle(&self) -> (Matrix<T, 3, 1>, T) {
        let q = self.normalize();
        let k = (T::one() - q.w() * q.w()).sqrt();
        if k < T::epsilon() {
            (
                Matrix::from([[T::one()], [T::zero()], [T::zero()]]),
                T::zero(),
            )
        } else {
            let vec_x = q.x() / k;
            let vec_y = q.y() / k;
            let vec_z = q.z() / k;
            (
                Matrix::from([[vec_x], [vec_y], [vec_z]]),
                T::from(2.).unwrap() * self.w().acos(),
            )
        }
    }
    ///rotation of a vector p, by a unit quaternion q:  q * p q', where q' is the conjugate
    #[allow(dead_code)]
    pub fn rotate_vector(&self, p: Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
        let quat_p = QuatR(Quat::init(p[[0, 0]], p[[1, 0]], p[[2, 0]], T::zero()));
        let temp2 = &(self * &quat_p) * &self.conjugate();
        Matrix::from([[temp2.x()], [temp2.y()], [temp2.z()]])
    }
}

#[test]
fn test_quatr_convert_axis_angle_0() {
    use more_asserts::assert_le;
    use std::f64::consts::PI;
    //convert axis angle to quaternion representation and back
    let axis = Matrix::from([[1., 2., 3.]]).t();

    let axis_normalize = axis.normalize_l2();
    let q = QuatR::init_from_axis_angle_degree(axis, 90.);
    let (a, angle) = q.to_axis_angle();
    assert_eq!(a, axis_normalize);
    assert_le!((angle / PI * 180. - 90.).abs(), 1e-9);
}
#[test]
fn test_quatr_convert_axis_angle_1() {
    use more_asserts::assert_le;
    use std::f32::consts::PI;
    //convert axis angle to quaternion representation and back
    let axis = Matrix::from([[1., 2., 3.]]).t();

    let axis_normalize = axis.normalize_l2();
    let q = QuatR::init_from_axis_angle_degree(axis, 370.);
    let (a, angle) = q.to_axis_angle();
    // assert_eq!(a, axis_normalize);
    assert_matrix_approx_eq_float(&a, &axis_normalize, 1e-5);
    assert_le!((angle / PI * 180. - 10.).abs(), 1e-4);
}
#[test]
fn test_quatr_convert_axis_angle_2() {
    use std::f32::consts::PI;
    //convert axis angle to quaternion representation and back
    let axis = Matrix::from([[1., 2., 3.]]).t();

    let axis_normalize = axis.normalize_l2();
    let q = QuatR::init_from_axis_angle_degree(axis, -33.);
    let (a, angle) = q.to_axis_angle();
    assert!(
        ((angle / PI * 180. - (360. - 33.)).abs() < 1e-9 && a == axis_normalize)
            || ((angle / PI * 180. + (360. - 33.)).abs() < 1e-9 && (a * -1.) == (axis_normalize))
    );
}
#[test]
fn test_quatr_rotate_point() {
    use more_asserts::assert_le;
    use std::f32::consts::PI;
    //compute rotation using quaternion
    //rotate a vector using the rotation matrix and compare to rotation using quaternions
    let p = Matrix::from([[1., 5., -3.]]).t();
    let axis = Matrix::from([[1., 0., 0.]]).t();

    let axis_normalize = axis.normalize_l2();
    let q = QuatR::init_from_axis_angle_degree(axis, 90.);
    let (a, angle) = q.to_axis_angle();
    assert_eq!(a, axis_normalize);
    assert_le!((angle / PI * 180. - 90.).abs(), 1e-5);

    let rot = q.to_matrix();

    let rot_expected = Matrix::from([
        [1., 0., 0., 0.],
        [0., 0., -1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
    ]);

    assert_eq!(rot, rot_expected);
    // println!("rot: {:#?}", rot);
    // println!("rot_expected: {:#?}", rot_expected);

    let vec_rotated_expected =
        rot_expected.dot(&Matrix::from([[p[[0, 0]], p[[1, 0]], p[[2, 0]], 1.]]).t());

    let vec_rotated = q.rotate_vector(p);

    assert_matrix_approx_eq_float(
        &vec_rotated,
        &Matrix::from([[
            vec_rotated_expected[[0, 0]],
            vec_rotated_expected[[1, 0]],
            vec_rotated_expected[[2, 0]],
        ]])
        .t(),
        1e-6,
    );
}
