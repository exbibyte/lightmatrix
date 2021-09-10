use crate::mat3::Mat3i;
use crate::mat3x1::Mat3x1i;

use num_traits::{float::Float, int::PrimInt, NumAssign};

pub trait Dot<Rhs> {
    type R;
    fn dot2(&self, rhs: &Rhs) -> Self::R;
}

impl<T: NumAssign + Copy> Dot<Mat3i<T>> for Mat3i<T> {
    type R = Mat3i<T>;
    fn dot2(&self, rhs: &Mat3i<T>) -> Self::R {
        Mat3i::from(self.0.dot(&rhs.0))
    }
}

impl<T: NumAssign + Copy> Dot<Mat3x1i<T>> for Mat3i<T> {
    type R = Mat3x1i<T>;
    fn dot2(&self, rhs: &Mat3x1i<T>) -> Self::R {
        Mat3x1i::from(self.0.dot_vec(&rhs.0))
    }
}

// pub fn dot2<Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> <Lhs as Dot<Rhs>>::R
// where
//     Lhs: Dot<Rhs>,
// {
//     lhs.dot2(rhs)
// }

#[test]
fn test_operator_dot() {
    let a = Mat3i::from([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Mat3i::from([[1i32, 2i32, 3i32], [4i32, 5i32, 6i32], [7i32, 8i32, 9i32]]);
    let expect = Mat3i::from([
        [18i32, 21i32, 24i32],
        [54i32, 66i32, 78i32],
        [90i32, 111i32, 132i32],
    ]);
    let ret = a.dot2(&b);
    assert_eq!(ret, expect);
}

#[test]
fn test_operator_dot_vec() {
    let a = Mat3i::from([[0i32, 1i32, 2i32], [3i32, 4i32, 5i32], [6i32, 7i32, 8i32]]);
    let b = Mat3x1i::from([[1i32], [4i32], [7i32]]);
    let expect = Mat3x1i::from([[18i32], [54i32], [90i32]]);
    assert_eq!(a.dot2(&b), expect);
}
