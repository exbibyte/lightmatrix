use num_traits::NumAssign;

pub struct Scalar<T: NumAssign + Copy + Default>(pub T);

impl<T: NumAssign + Copy + Default> From<T> for Scalar<T> {
    fn from(x: T) -> Self {
        Self(x)
    }
}
