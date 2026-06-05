use crate::simd;
use std::ops::{Add, Div, Mul, Sub};
pub trait Scalar:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sized
    + Copy
    + Default
    + PartialOrd
{
    fn vector_add(left: &[Self], right: &[Self]) -> Vec<Self> {
        // NOTE: Benchmarks show that this zero-init has no cost,
        // Likely due to LLVM detecting them as dead-stores
        let mut out = vec![Self::default(); left.len()];
        Self::vector_add_to_out(left, right, &mut out);
        out
    }
    fn vector_add_to_out(left: &[Self], right: &[Self], out: &mut [Self]) {
        vector_op_fallback_to_out(left, right, out, |l, r| l + r);
    }
    fn vector_add_inplace(left: &mut [Self], right: &[Self]) {
        vector_op_fallback_inplace(left, right, |l, r| l + r);
    }
}
impl Scalar for f64 {
    fn vector_add_to_out(left: &[Self], right: &[Self], out: &mut [Self]) {
        simd::add_to_out(left, right, out);
    }
    fn vector_add_inplace(left: &mut [Self], right: &[Self]) {
        simd::add_inplace(left, right);
    }
}

#[inline(always)]
pub fn vector_op_fallback_to_out<V, OP>(left: &[V], right: &[V], out: &mut [V], op: OP)
where
    V: Scalar,
    OP: Fn(V, V) -> V,
{
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), out.len());
    left.iter()
        .zip(right)
        .zip(out)
        .for_each(|((x, y), e)| *e = op(*x, *y));
}

#[inline(always)]
fn vector_op_fallback_inplace<V, OP>(left: &mut [V], right: &[V], op: OP)
where
    V: Scalar,
    OP: Fn(V, V) -> V,
{
    left.iter_mut()
        .zip(right)
        .for_each(|(l, r)| *l = op(*l, *r));
}

pub fn vector_add<V, A, B>(left: &A, right: &B) -> Vec<V>
where
    V: Scalar,
    A: AsRef<[V]>,
    B: AsRef<[V]>,
{
    V::vector_add(left.as_ref(), right.as_ref())
}
