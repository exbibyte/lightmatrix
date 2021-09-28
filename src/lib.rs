#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[macro_use]
pub mod matrix;
pub mod dualquat;
pub mod dualscalar;
pub mod matrix_slice;
pub mod operator;
pub mod quat;
pub mod quatr;
pub mod quatt;
pub mod scalar;
