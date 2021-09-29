#[allow(unused_imports)]
use std::ops::Div;
#[allow(unused_imports)]
use std::ops::Index;
#[allow(unused_imports)]
use std::ops::IndexMut;
#[allow(unused_imports)]
use std::ops::{Add, Mul, Sub};

use num_traits::{float::Float, real::Real, NumAssign};

use crate::dualscalar::*;
use crate::matrix::*;
use crate::quat::*;
use crate::quatr::*;
use crate::quatt::*;
// use constants::*;

/// represents rot, translation pair
#[derive(Clone, Default, Debug)]
pub struct DualQuat<T: Float + NumAssign + Default + Clone + PartialEq>(QuatR<T>, QuatT<T>);

macro_rules! impl_dualquat_partialeq_float {
    ($type:ty) => {
        impl PartialEq<DualQuat<$type>> for DualQuat<$type> {
            fn eq(&self, other: &Self) -> bool {
                (&self.0).eq(&other.0) && (&self.1).eq(&other.1)
            }
        }
    };
}

impl_dualquat_partialeq_float!(f32);
impl_dualquat_partialeq_float!(f64);

impl<T: Float + Default + NumAssign> DualQuat<T> {
    #[allow(dead_code)]
    pub fn quat_rot(&self) -> &QuatR<T> {
        &self.0
    }
    #[allow(dead_code)]
    pub fn quat_tra(&self) -> &QuatT<T> {
        &self.1
    }
    pub fn quat_rot_mut(&mut self) -> &mut QuatR<T> {
        &mut self.0
    }
    #[allow(dead_code)]
    pub fn quat_tra_mut(&mut self) -> &mut QuatT<T> {
        &mut self.1
    }
    #[allow(dead_code)]
    pub fn dual_scalar(&self) -> DualScalar<T> {
        DualScalar::new(self.quat_rot().w(), self.quat_tra().w())
    }
    #[allow(dead_code)]
    pub fn init_from_rot(rotate: QuatR<T>) -> DualQuat<T> {
        DualQuat(
            rotate.normalize(),
            QuatT::init(T::zero(), T::zero(), T::zero(), T::zero()),
        )
    }
    #[allow(dead_code)]
    pub fn init_from_tra(translate: QuatT<T>) -> DualQuat<T> {
        DualQuat(
            QuatR::init(T::zero(), T::zero(), T::zero(), T::one()),
            translate,
        )
    }
    #[allow(dead_code)]
    pub fn init(rotate: QuatR<T>, translate: QuatT<T>) -> DualQuat<T> {
        let t = &translate * &rotate;
        DualQuat(rotate, QuatT(t))
    }
    #[allow(dead_code)]
    pub fn init_raw(rotate: QuatR<T>, translate: QuatT<T>) -> DualQuat<T> {
        DualQuat(rotate, translate)
    }
    /// returns 4x4 homogeneous matrix
    pub fn xform_rot(&self) -> Matrix<T, 4, 4> {
        self.normalize().quat_rot().to_matrix()
    }
    /// returns 4x1 translation vector
    pub fn xform_tra(&self) -> Matrix<T, 4, 1> {
        let a = self.normalize();
        let b = &(a.quat_tra() * T::from(2.).unwrap()) * &a.quat_rot().conjugate();
        Matrix::from([[b.x(), b.y(), b.z(), T::one()]]).t()
    }
    ///returns 4x4 homogeneous matrix
    pub fn xform(&self) -> Matrix<T, 4, 4> {
        let a = self.xform_tra();
        let mut b = self.xform_rot();
        b[[0, 3]] = a[[0, 0]];
        b[[1, 3]] = a[[1, 0]];
        b[[2, 3]] = a[[2, 0]];
        b
    }
    pub fn normalize(&self) -> Self {
        // dbg!(self);
        let l = self.quat_rot().norm();
        assert!(l > T::epsilon());
        let a = self.quat_rot().scale(T::one() / l);
        let b = self.quat_tra().scale(T::one() / l);
        DualQuat(a, b)
    }
    pub fn normalized(&mut self) {
        let (a, b) = <(QuatR<T>, QuatT<T>)>::from(self.normalize());
        *self.quat_rot_mut() = a;
        *self.quat_tra_mut() = b;
    }
    pub fn norm(&self) -> DualScalar<T> {
        DualScalar::new(
            self.quat_rot().norm(),
            self.quat_rot().0.dot(&self.quat_tra().0) / self.quat_rot().norm(),
        )
    }
    pub fn conjugate(&self) -> Self {
        DualQuat(self.quat_rot().conjugate(), self.quat_tra().conjugate())
    }
    pub fn rotate_point(&self, p: Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
        let t =
            &(self * &DualQuat::init_from_tra(QuatT::init_from_translation(p))) * &self.conjugate();
        let v = t.xform_tra();
        Matrix::from([[v[[0, 0]], v[[1, 0]], v[[2, 0]]]]).t()
    }
    /// q(q* p)^t
    pub fn sclerp(&self, other: &Self, t: T) -> Self {
        // self * (&(&self.conjugate() * other).pow(t))

        // take shortest
        let dot = self.quat_rot().dot(other.quat_rot());
        let o;
        if dot < T::zero() {
            o = DualQuat::init_raw(other.quat_rot() * -T::one(), other.quat_tra() * -T::one());
        } else {
            o = other.clone();
        }

        let diff = &self.conjugate() * &o;

        self * &diff.pow(t)
    }
    fn pow(&self, e: T) -> Self {
        let vr = Matrix::from([[
            self.quat_rot().x(),
            self.quat_rot().y(),
            self.quat_rot().z(),
        ]])
        .t();

        let vd = Matrix::from([[
            self.quat_tra().x(),
            self.quat_tra().y(),
            self.quat_tra().z(),
        ]])
        .t();

        let invr = T::one() / vr.norm_l2();
        // dbg!(&diff);
        // dbg!(&invr);
        if invr.is_infinite() {
            //pure translation
            let mut q = self.clone();
            *q.quat_tra_mut().x_mut() = q.quat_tra().x() * e;
            *q.quat_tra_mut().y_mut() = q.quat_tra().y() * e;
            *q.quat_tra_mut().z_mut() = q.quat_tra().z() * e;
            return q.normalize();
        }

        // screw
        let mut angle = T::from(2.).unwrap() * self.quat_rot().w().acos();
        let mut pitch = T::from(2.).unwrap().neg() * self.quat_tra().w() * invr;
        let direction = vr * invr;

        // dbg!(&direction);
        let moment =
            (vd - ((direction * pitch) * (self.quat_rot().w() * T::from(0.5).unwrap()))) * invr;

        angle *= e;
        pitch *= e;

        //convert back
        let sin_angle = (T::from(0.5).unwrap() * angle).sin();
        let cos_angle = (T::from(0.5).unwrap() * angle).cos();
        let temp = direction * sin_angle;
        let real = QuatR::init(temp[[0, 0]], temp[[1, 0]], temp[[2, 0]], cos_angle);

        let temp2 = (moment * sin_angle) + (direction * pitch * T::from(0.5).unwrap() * cos_angle);
        let dual = QuatT::init(
            temp2[[0, 0]],
            temp2[[1, 0]],
            temp2[[2, 0]],
            -pitch * T::from(0.5).unwrap() * sin_angle,
        );

        DualQuat(real, dual)
    }
    // ///todo: debug this function
    // fn pow(&self, e: T) -> DualQuat {
    //     let mut d = self.clone();

    //     let mut screwaxis = Matrix1D::from(arr1(&[0., 0., 0.]));
    //     let mut moment = Matrix1D::from(arr1(&[0., 0., 0.]));
    //     let mut angles = Matrix1D::from(arr1(&[0., 0.]));

    //     let norm_a = d.get_screw_parameters(&mut screwaxis, &mut moment, &mut angles);

    //     // pure translation
    //     if norm_a < EPS {
    //         println!("pure translation");
    //         *d.quat_tra_mut().x_mut() = d.quat_tra().x() * e;
    //         *d.quat_tra_mut().y_mut() = d.quat_tra().y() * e;
    //         *d.quat_tra_mut().z_mut() = d.quat_tra().z() * e;
    //         d.normalized();
    //         d
    //     } else {
    //         // exponentiate
    //         let theta = angles[0] * e;
    //         let alpha = angles[1] * e;
    //         println!("theta, alpha: {:?}, {:?}", &theta, &alpha);
    //         // convert back
    //         d.set_screw_parameters(screwaxis.view(), moment.view(), theta, alpha);
    //         d
    //     }
    // }
    pub fn get_screw_parameters(
        &self,
        screwaxis: &mut Matrix<T, 3, 1>,
        moment: &mut Matrix<T, 3, 1>,
        angles: &mut Matrix<T, 3, 1>,
    ) -> T {
        let q_a = Matrix::from([[
            self.quat_rot().x(),
            self.quat_rot().y(),
            self.quat_rot().z(),
        ]])
        .t();

        let q_b = Matrix::from([[
            self.quat_tra().x(),
            self.quat_tra().y(),
            self.quat_tra().z(),
        ]])
        .t();

        let norm_a = q_a.norm_l2();

        // pure translation
        if norm_a < T::epsilon() {
            println!("pure translation");
            let norm_a = q_b.norm_l2();
            *screwaxis = q_b.normalize_l2();

            for i in 0..3 {
                moment[[i, 0]] = T::zero();
            }
            angles[[0, 0]] = T::zero();
            angles[[1, 0]] = T::from(2.).unwrap() * q_b.norm_l2();
            norm_a
        } else {
            *screwaxis = q_a.normalize_l2();
            angles[[0, 0]] = T::from(2.).unwrap() * norm_a.atan2(self.quat_rot().w());
            // if (angles[0] > Math.PI / 2) {
            //    angles[0] -= Math.PI;
            // }
            angles[[1, 0]] = T::from(2.).unwrap().neg() * self.quat_tra().w() / norm_a;
            let m1 = q_b / norm_a;
            let m2 = *screwaxis * self.quat_rot().w() * self.quat_tra().w() / (norm_a * norm_a);
            *moment = m1 + m2;
            norm_a
        }
    }

    pub fn set_screw_parameters(
        &mut self,
        screwaxis: Matrix<T, 3, 1>,
        moment: Matrix<T, 3, 1>,
        theta: T,
        alpha: T,
    ) {
        let two = T::from(2.).unwrap();

        let cosa = (theta / two).cos();
        let sina = (theta / two).sin();

        *self.quat_rot_mut().w_mut() = cosa;
        *self.quat_rot_mut().x_mut() = sina * screwaxis[[0, 0]];
        *self.quat_rot_mut().x_mut() = sina * screwaxis[[1, 0]];
        *self.quat_rot_mut().x_mut() = sina * screwaxis[[2, 0]];

        *self.quat_tra_mut().w_mut() = -alpha / two * sina;
        *self.quat_tra_mut().x_mut() =
            sina * moment[[0, 0]] + alpha / two * cosa * screwaxis[[0, 0]];
        *self.quat_tra_mut().y_mut() =
            sina * moment[[1, 0]] + alpha / two * cosa * screwaxis[[1, 0]];
        *self.quat_tra_mut().z_mut() =
            sina * moment[[2, 0]] + alpha / two * cosa * screwaxis[[2, 0]];

        self.normalized();
    }
}

///useful for transforms, eg: p_new = q*p*q', q := transform, q' := conjugate, p := vector point in dualquat
impl<T: Float + Default + NumAssign> Mul for &DualQuat<T> {
    type Output = DualQuat<T>;
    fn mul(self, rhs: &DualQuat<T>) -> DualQuat<T> {
        let a = self;
        let b = rhs;
        DualQuat(
            a.quat_rot().mul(b.quat_rot()),
            QuatT(a.quat_tra() * b.quat_rot() + a.quat_rot() * b.quat_tra()),
        )
    }
}

impl<T: Float + Default + NumAssign> Add for &DualQuat<T> {
    type Output = DualQuat<T>;
    fn add(self, rhs: &DualQuat<T>) -> DualQuat<T> {
        DualQuat(
            self.quat_rot() + rhs.quat_rot(),
            self.quat_tra() + rhs.quat_tra(),
        )
    }
}

impl<T: Float + Default + NumAssign> Sub for &DualQuat<T> {
    type Output = DualQuat<T>;
    fn sub(self, rhs: &DualQuat<T>) -> DualQuat<T> {
        DualQuat(
            self.quat_rot() - rhs.quat_rot(),
            self.quat_tra() - rhs.quat_tra(),
        )
    }
}

impl<T: Float + Default + NumAssign> From<DualQuat<T>> for (QuatR<T>, QuatT<T>) {
    fn from(i: DualQuat<T>) -> (QuatR<T>, QuatT<T>) {
        (i.0, i.1)
    }
}

impl<T: Real + Default + NumAssign> Mul<&QuatT<T>> for &QuatR<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: &QuatT<T>) -> Quat<T> {
        &self.0 * &rhs.0
    }
}

impl<T: Real + Default + NumAssign> Mul<&QuatR<T>> for &QuatT<T> {
    type Output = Quat<T>;
    fn mul(self, rhs: &QuatR<T>) -> Quat<T> {
        &self.0 * &rhs.0
    }
}

impl<T: Real + Default + NumAssign> Add<&QuatT<T>> for &QuatR<T> {
    type Output = Quat<T>;
    fn add(self, rhs: &QuatT<T>) -> Quat<T> {
        &self.0 + &rhs.0
    }
}

impl<T: Real + Default + NumAssign> Add<&QuatR<T>> for &QuatT<T> {
    type Output = Quat<T>;
    fn add(self, rhs: &QuatR<T>) -> Quat<T> {
        &self.0 + &rhs.0
    }
}
