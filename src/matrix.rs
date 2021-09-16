use num_traits::{float::Float, int::PrimInt, NumAssign};

use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

#[derive(Debug)]
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

macro_rules! impl_partialeq_integral {
    ($type:ty) => {
        impl<const ROW: usize, const COL: usize> PartialEq<Matrix<$type, ROW, COL>>
            for Matrix<$type, ROW, COL>
        {
            fn eq(&self, other: &Self) -> bool {
                for i in 0..self.0.len() {
                    for j in 0..self.0[0].len() {
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
                for i in 0..self.0.len() {
                    for j in 0..self.0[0].len() {
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
impl_partialeq_float!(f32);
impl_partialeq_float!(f64);

impl<const ROW: usize, const COL: usize> Eq for Matrix<i8, ROW, COL> {}
impl<const ROW: usize, const COL: usize> Eq for Matrix<i32, ROW, COL> {}
impl<const ROW: usize, const COL: usize> Eq for Matrix<i64, ROW, COL> {}
