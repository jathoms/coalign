use crate::simd;
use paste::paste;
use std::ops::{Add, Div, Mul, Sub};

macro_rules! scalar_ops {
    ($($name:ident => $op:tt),*) => {
        paste! {
            $(
                fn [<vector_ $name>](left: &[Self], right: &[Self]) -> Vec<Self> {
                    let mut out = vec![Self::default(); left.len()];
                    Self::[<vector_ $name _to_out>](left, right, &mut out);
                    out
                }
                fn [<vector_ $name _to_out>](left: &[Self], right: &[Self], out: &mut [Self]) {
                    vector_op_fallback_to_out(left, right, out, |l, r| l $op r);
                }
                fn [<vector_ $name _inplace>](left: &mut [Self], right: &[Self]) {
                    vector_op_fallback_inplace(left, right, |l, r| l $op r);
                }
            )*
        }
    };
}

pub trait Scalar:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Copy
    + Default
    + PartialOrd
{
    scalar_ops!(add => +, sub => -, mul => *, div => /);
}

macro_rules! simd_overrides {
    ($($t:ident => $($name:ident),*);* $(;)?) => {
        paste! {
            $(
            impl Scalar for $t {
                $(
                fn [<vector_ $name _to_out>](left: &[Self], right: &[Self], out: &mut [Self]) {
                    simd::[<$name _to_out_ $t>](left, right, out);
                }
                fn [<vector_ $name _inplace>](left: &mut [Self], right: &[Self]) {
                    simd::[<$name _inplace_ $t>](left, right);
                }
                )*
            }
            )*
        }
    };
}

simd_overrides!(
    f64 => add, sub, mul, div;
    f32 => add, sub, mul, div;
    u8 => add, sub;
    u16 => add, sub, mul;
    u32 => add, sub, mul;
    u64 => add, sub, mul;
    i8 => add, sub;
    i16 => add, sub, mul;
    i32 => add, sub, mul;
    i64 => add, sub, mul;
);

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
pub fn vector_sub<V, A, B>(left: &A, right: &B) -> Vec<V>
where
    V: Scalar,
    A: AsRef<[V]>,
    B: AsRef<[V]>,
{
    V::vector_sub(left.as_ref(), right.as_ref())
}
pub fn vector_mul<V, A, B>(left: &A, right: &B) -> Vec<V>
where
    V: Scalar,
    A: AsRef<[V]>,
    B: AsRef<[V]>,
{
    V::vector_mul(left.as_ref(), right.as_ref())
}
pub fn vector_div<V, A, B>(left: &A, right: &B) -> Vec<V>
where
    V: Scalar,
    A: AsRef<[V]>,
    B: AsRef<[V]>,
{
    V::vector_div(left.as_ref(), right.as_ref())
}
