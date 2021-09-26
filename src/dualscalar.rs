use std::ops::{Add, Div, Mul, Sub};

use num_traits::{real::Real, NumAssign};

///(real, img) pair where img*img=0, img~=0
#[derive(Debug, Clone, PartialEq)]
pub struct DualScalar<T: Real + NumAssign + Default + Clone + PartialEq>(T, T);

impl<T: Real + Default + NumAssign> DualScalar<T> {
    pub fn new(real: T, img: T) -> Self {
        Self(real, img)
    }
    pub fn real(&self) -> T {
        self.0
    }
    pub fn dual(&self) -> T {
        self.1
    }
    pub fn real_mut(&mut self) -> &mut T {
        &mut self.0
    }
    pub fn dual_mut(&mut self) -> &mut T {
        &mut self.1
    }
    pub fn conjugate(&self) -> Self {
        Self::new(self.real(), -self.dual())
    }
    pub fn invert(&self) -> Self {
        let a = T::one() / self.real();
        let b = -self.dual() * a * a;
        Self::new(a, b)
    }
    pub fn norm(&self) -> T {
        self.real()
    }
    pub fn pow(&self, e: Self) -> Self {
        let a = self.real().powf(e.real());
        let b = self.dual() / self.real() * e.real() * a + e.dual() * a * a.ln();
        Self::new(a, b)
    }
    pub fn sqrt(&self) -> Self {
        let a = self.real().sqrt();
        let b = self.dual() / (T::from(2.).unwrap() * a);
        Self::new(a, b)
    }
}

impl<T: Real + Default + NumAssign> Add for DualScalar<T> {
    type Output = DualScalar<T>;
    fn add(self, rhs: DualScalar<T>) -> Self::Output {
        DualScalar::new(self.real() + rhs.real(), self.dual() + rhs.dual())
    }
}
impl<T: Real + Default + NumAssign> Sub for DualScalar<T> {
    type Output = DualScalar<T>;
    fn sub(self, rhs: DualScalar<T>) -> Self::Output {
        DualScalar::new(self.real() - rhs.real(), self.dual() - rhs.dual())
    }
}
impl<T: Real + Default + NumAssign> Mul for DualScalar<T> {
    type Output = DualScalar<T>;
    fn mul(self, rhs: DualScalar<T>) -> Self::Output {
        DualScalar::new(
            self.real() * rhs.real(),
            self.real() * rhs.dual() + self.dual() * rhs.real(),
        )
    }
}
impl<T: Real + Default + NumAssign> Div for DualScalar<T> {
    type Output = DualScalar<T>;
    fn div(self, rhs: DualScalar<T>) -> Self::Output {
        DualScalar::new(
            self.real() / rhs.real(),
            (rhs.real() * self.dual() - self.real() * rhs.dual()) / (self.real() * rhs.real()),
        )
    }
}
