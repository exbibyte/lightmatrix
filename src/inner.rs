use num_traits::{float::Float, int::PrimInt, NumAssign};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

macro_rules! delegate_num_ops_inner {
    ($struct_name:ident, $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> $trait
            for $struct_name<T, ROW, COL>
        {
            type Output = Self;

            fn $func(self, rhs: Self) -> Self {
                let mut ret = Self::default();

                for y in 0..self.array.len() {
                    for x in 0..self.array[0].len() {
                        ret.array[y][x] = self.array[y][x].$func(rhs.array[y][x]);
                    }
                }

                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_inner {
    ($struct_name:ident, $trait:ident, $func:ident) => {
        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> $trait
            for $struct_name<T, ROW, COL>
        {
            fn $func(&mut self, rhs: Self) {
                for y in 0..self.array.len() {
                    for x in 0..self.array[0].len() {
                        self.array[y][x].$func(rhs.array[y][x]);
                    }
                }
            }
        }
    };
}

macro_rules! delegate_num_ops_inner_scalar {
    ($struct_name:ident, $trait:ident, $func:ident, $scalar_type:ty) => {
        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> $trait<$scalar_type>
            for $struct_name<T, ROW, COL>
        {
            type Output = Self;

            fn $func(self, rhs: $scalar_type) -> Self {
                let mut ret = Self::default();

                for y in 0..self.array.len() {
                    for x in 0..self.array[0].len() {
                        ret.array[y][x] = self.array[y][x].$func(rhs);
                    }
                }

                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_inner_scalar {
    ($struct_name:ident, $trait:ident, $func:ident, $scalar_type:ty) => {
        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> $trait<$scalar_type>
            for $struct_name<T, ROW, COL>
        {
            fn $func(&mut self, rhs: $scalar_type) {
                for y in 0..self.array.len() {
                    for x in 0..self.array[0].len() {
                        self.array[y][x].$func(rhs);
                    }
                }
            }
        }
    };
}

macro_rules! create_inner_struct {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name<T: NumAssign + Copy, const ROW: usize, const COL: usize> {
            array: [[T; COL]; ROW],
        }

        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> Default
            for $name<T, ROW, COL>
        {
            fn default() -> Self {
                Self {
                    array: [[T::zero(); COL]; ROW],
                }
            }
        }

        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> From<[[T; COL]; ROW]>
            for $name<T, ROW, COL>
        {
            fn from(src: [[T; COL]; ROW]) -> Self {
                $name { array: src }
            }
        }

        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> Deref for $name<T, ROW, COL> {
            type Target = [[T; COL]; ROW];

            fn deref(&self) -> &Self::Target {
                &self.array
            }
        }

        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> DerefMut
            for $name<T, ROW, COL>
        {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.array
            }
        }
        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> Index<[usize; 2]>
            for $name<T, ROW, COL>
        {
            type Output = T;

            fn index(&self, index: [usize; 2]) -> &Self::Output {
                &self.array[index[0]][index[1]]
            }
        }

        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> IndexMut<[usize; 2]>
            for $name<T, ROW, COL>
        {
            fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
                &mut self.array[index[0]][index[1]]
            }
        }
        impl<T: NumAssign + Copy, const ROW: usize, const COL: usize> $name<T, ROW, COL> {
            pub fn t(&self) -> $name<T, COL, ROW> {
                let mut ret = $name::<T, COL, ROW>::default();
                for i in 0..self.array.len() {
                    for j in 0..self.array[0].len() {
                        ret.array[j][i] = self.array[i][j];
                    }
                }
                ret
            }
            pub fn dot(&self, other: &$name<T, COL, ROW>) -> $name<T, ROW, ROW> {
                let mut ret = $name::<T, ROW, ROW>::default();
                for i in 0..ROW {
                    for j in 0..ROW {
                        for k in 0..COL {
                            ret.array[i][j] += self.array[i][k] * other.array[k][j];
                        }
                    }
                }
                ret
            }
            pub fn dot_vec(&self, other: &$name<T, COL, 1>) -> $name<T, ROW, 1> {
                let mut ret = $name::<T, ROW, 1>::default();
                for i in 0..ROW {
                    for j in 0..COL {
                        ret.array[i][0] += self.array[i][j] * other.array[j][0];
                    }
                }
                ret
            }
        }

        delegate_num_ops_inner!($name, Mul, mul);
        delegate_num_ops_inner!($name, Div, div);
        delegate_num_ops_inner!($name, Add, add);
        delegate_num_ops_inner!($name, Sub, sub);
        delegate_num_ops_assign_inner!($name, SubAssign, sub_assign);
        delegate_num_ops_assign_inner!($name, AddAssign, add_assign);
        delegate_num_ops_assign_inner!($name, MulAssign, mul_assign);
        delegate_num_ops_assign_inner!($name, DivAssign, div_assign);

        delegate_num_ops_inner_scalar!($name, Mul, mul, T);
        delegate_num_ops_inner_scalar!($name, Div, div, T);
        delegate_num_ops_inner_scalar!($name, Add, add, T);
        delegate_num_ops_inner_scalar!($name, Sub, sub, T);
        delegate_num_ops_assign_inner_scalar!($name, SubAssign, sub_assign, T);
        delegate_num_ops_assign_inner_scalar!($name, AddAssign, add_assign, T);
        delegate_num_ops_assign_inner_scalar!($name, MulAssign, mul_assign, T);
        delegate_num_ops_assign_inner_scalar!($name, DivAssign, div_assign, T);
    };
}

create_inner_struct!(InnerInt);

impl<T: PrimInt + NumAssign, const ROW: usize, const COL: usize> PartialEq
    for InnerInt<T, ROW, COL>
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.array.len() {
            for j in 0..self.array[0].len() {
                if self.array[i][j] != other.array[i][j] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: PrimInt + NumAssign, const ROW: usize, const COL: usize> Eq for InnerInt<T, ROW, COL> {}

create_inner_struct!(InnerFloat);

impl<T: Float + NumAssign, const ROW: usize, const COL: usize> PartialEq
    for InnerFloat<T, ROW, COL>
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.array.len() {
            for j in 0..self.array[0].len() {
                if !self.array[i][j].is_finite() || !other.array[i][j].is_finite() {
                    return false;
                } else if (self.array[i][j] - other.array[i][j]).abs() > Float::epsilon() {
                    return false;
                }
            }
        }
        true
    }
}
