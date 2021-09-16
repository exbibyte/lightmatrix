use num_traits::{float::Float, int::PrimInt, NumAssign};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

use crate::matrix::Matrix;

pub trait Dot<Rhs> {
    type R;
    fn dot(&self, rhs: &Rhs) -> Self::R;
}

// pub fn dot2<Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> <Lhs as Dot<Rhs>>::R
// where
//     Lhs: Dot<Rhs>,
// {
//     lhs.dot2(rhs)
// }

impl<T: NumAssign + Copy + Default, const ROW_1: usize, const COL_1: usize, const COL_2: usize>
    Dot<Matrix<T, COL_1, COL_2>> for Matrix<T, ROW_1, COL_1>
{
    type R = Matrix<T, ROW_1, COL_2>;
    fn dot(&self, other: &Matrix<T, COL_1, COL_2>) -> Matrix<T, ROW_1, COL_2> {
        let mut ret = Matrix([[T::default(); COL_2]; ROW_1]);
        for i in 0..ROW_1 {
            for j in 0..COL_2 {
                for k in 0..COL_1 {
                    ret[[i, j]] += self[[i, k]] * other[[k, j]];
                }
            }
        }
        ret
    }
}

macro_rules! delegate_num_ops {
    ($struct_name:ident, $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait
            for $struct_name<T, ROW, COL>
        {
            type Output = Self;

            fn $func(self, rhs: Self) -> Self {
                let mut ret = $struct_name([[T::default(); COL]; ROW]);

                for y in 0..ROW {
                    for x in 0..COL {
                        ret[[y, x]] = self[[y, x]].$func(rhs[[y, x]]);
                    }
                }

                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign {
    ($struct_name:ident, $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait
            for $struct_name<T, ROW, COL>
        {
            fn $func(&mut self, rhs: Self) {
                for y in 0..ROW {
                    for x in 0..COL {
                        self[[y, x]].$func(rhs[[y, x]]);
                    }
                }
            }
        }
    };
}

macro_rules! delegate_num_ops_scalar {
    ($struct_name:ident, $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait<T>
            for $struct_name<T, ROW, COL>
        {
            type Output = Self;

            fn $func(self, rhs: T) -> Self {
                let mut ret = Self::default();

                for y in 0..ROW {
                    for x in 0..COL {
                        ret[[y, x]] = self[[y, x]].$func(rhs);
                    }
                }

                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_scalar {
    ($struct_name:ident, $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy + Default, const ROW: usize, const COL: usize> $trait<T>
            for $struct_name<T, ROW, COL>
        {
            fn $func(&mut self, rhs: T) {
                for y in 0..ROW {
                    for x in 0..COL {
                        self[[y, x]].$func(rhs);
                    }
                }
            }
        }
    };
}

delegate_num_ops!(Matrix, Mul, mul);
delegate_num_ops!(Matrix, Div, div);
delegate_num_ops!(Matrix, Add, add);
delegate_num_ops!(Matrix, Sub, sub);

delegate_num_ops_assign!(Matrix, MulAssign, mul_assign);
delegate_num_ops_assign!(Matrix, DivAssign, div_assign);
delegate_num_ops_assign!(Matrix, AddAssign, add_assign);
delegate_num_ops_assign!(Matrix, SubAssign, sub_assign);

delegate_num_ops_scalar!(Matrix, Mul, mul);
delegate_num_ops_scalar!(Matrix, Div, div);
delegate_num_ops_scalar!(Matrix, Add, add);
delegate_num_ops_scalar!(Matrix, Sub, sub);

delegate_num_ops_assign_scalar!(Matrix, SubAssign, sub_assign);
delegate_num_ops_assign_scalar!(Matrix, AddAssign, add_assign);
delegate_num_ops_assign_scalar!(Matrix, MulAssign, mul_assign);
delegate_num_ops_assign_scalar!(Matrix, DivAssign, div_assign);

#[test]
fn test_rawarray_operator_dot() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Matrix([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Matrix([
        [18i32, 21i32, 24i32],
        [54i32, 66i32, 78i32],
        [90i32, 111i32, 132i32],
    ]);
    let ret = a.dot(&b);
    assert_eq!(ret, expect);
}

#[test]
fn test_rawarray_operator_dot_vec() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Matrix([[1i32], [4i32], [7i32]]);
    let expect = Matrix([[18i32], [54i32], [90i32]]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_rawarray_operator_dot_vec_float() {
    let a = Matrix([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = Matrix([[1f32], [4f32], [7f32]]);
    let expect = Matrix([[18f32], [54f32], [90f32]]);
    assert_eq!(a.dot(&b), expect);
}

#[test]
fn test_rawarray_operator_mul() {
    let a = Matrix([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Matrix([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Matrix([
        [0i32, 2i32, 6i32],
        [12i32, 20i32, 30i32],
        [42i32, 56i32, 72i32],
    ]);
    let ret = a * b;
    assert_eq!(ret, expect);
}

#[test]
fn test_rawarray_operator_mul_scalar() {
    let a = Matrix([[0f32, 1f32, 2f32], [3f32, 4f32, 5f32], [6f32, 7f32, 8f32]]);
    let b = 6f32;
    let expect = Matrix([
        [0f32, 6f32, 12f32],
        [18f32, 24f32, 30f32],
        [36f32, 42f32, 48f32],
    ]);
    let ret = a * b;
    assert_eq!(ret, expect);
}
