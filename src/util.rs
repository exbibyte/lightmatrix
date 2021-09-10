macro_rules! delegate_num_ops_mat {
    ($trait:ident, $func:ident, $mat:ident) => {
        impl<T: NumAssign + Copy> $trait for $mat<T> {
            type Output = Self;

            fn $func(self, rhs: Self) -> Self {
                let ret = $mat::from(self.0.$func(rhs.0));
                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_mat {
    ($trait:ident, $func:ident, $mat:ident) => {
        impl<T: NumAssign + Copy> $trait for $mat<T> {
            fn $func(&mut self, rhs: Self) {
                self.0.$func(rhs.0);
            }
        }
    };
}

macro_rules! delegate_num_ops_scalar_mat {
    ($trait:ident, $func:ident, $mat:ident) => {
        impl<T: NumAssign + Copy> $trait<T> for $mat<T> {
            type Output = Self;

            fn $func(self, rhs: T) -> Self {
                let ret = $mat::from(self.0.$func(rhs));
                ret
            }
        }
    };
}

macro_rules! delegate_num_ops_assign_scalar_mat {
    ($trait:ident, $func:ident, $mat:ident) => {
        impl<T: NumAssign + Copy> $trait<T> for $mat<T> {
            fn $func(&mut self, rhs: T) {
                self.0.$func(rhs);
            }
        }
    };
}

macro_rules! create_matrix {
    ($matrix_name:ident, $inner_name:ident, $rows:expr, $cols:expr) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $matrix_name<T: NumAssign + Copy>(pub(crate) $inner_name<T, $rows, $cols>);

        impl<T: NumAssign + Copy> Deref for $matrix_name<T> {
            type Target = $inner_name<T, $rows, $cols>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<T: NumAssign + Copy> DerefMut for $matrix_name<T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<T: NumAssign + Copy> From<$inner_name<T, $rows, $cols>> for $matrix_name<T> {
            fn from(inner: $inner_name<T, $rows, $cols>) -> Self {
                Self(inner)
            }
        }

        impl<T: NumAssign + Copy> From<[[T; $cols]; $rows]> for $matrix_name<T> {
            fn from(src: [[T; $cols]; $rows]) -> Self {
                Self($inner_name::from(src))
            }
        }

        impl<T: NumAssign + Copy> $matrix_name<T> {
            pub fn zero() -> Self {
                Self($inner_name::from([[T::zero(); $cols]; $rows]))
            }
        }

        delegate_num_ops_mat!(Add, add, $matrix_name);
        delegate_num_ops_mat!(Sub, sub, $matrix_name);
        delegate_num_ops_mat!(Mul, mul, $matrix_name);
        delegate_num_ops_mat!(Div, div, $matrix_name);
        delegate_num_ops_assign_mat!(AddAssign, add_assign, $matrix_name);
        delegate_num_ops_assign_mat!(SubAssign, sub_assign, $matrix_name);
        delegate_num_ops_assign_mat!(MulAssign, mul_assign, $matrix_name);
        delegate_num_ops_assign_mat!(DivAssign, div_assign, $matrix_name);

        delegate_num_ops_scalar_mat!(Add, add, $matrix_name);
        delegate_num_ops_scalar_mat!(Sub, sub, $matrix_name);
        delegate_num_ops_scalar_mat!(Mul, mul, $matrix_name);
        delegate_num_ops_scalar_mat!(Div, div, $matrix_name);
        delegate_num_ops_assign_scalar_mat!(AddAssign, add_assign, $matrix_name);
        delegate_num_ops_assign_scalar_mat!(SubAssign, sub_assign, $matrix_name);
        delegate_num_ops_assign_scalar_mat!(MulAssign, mul_assign, $matrix_name);
        delegate_num_ops_assign_scalar_mat!(DivAssign, div_assign, $matrix_name);
    };
}
