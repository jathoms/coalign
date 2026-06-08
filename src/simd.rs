use paste::paste;
use pulp::Simd;

macro_rules! impl_simd_ops {
    ($t:ident, $($name:ident => $simd_method:ident => $op:tt),*) => {
        paste! {
            $(
            #[pulp::with_simd([<$name _to_out_ $t>] = pulp::Arch::new())]
            #[inline(always)]
            pub fn [<$name _to_out_impl_ $t>]<S: Simd>(simd: S, left: &[$t], right: &[$t], out: &mut [$t]) {
                let (left_head, left_tail) = S::[<as_simd_ $t s>](left);
                let (right_head, right_tail) = S::[<as_simd_ $t s>](right);
                let (out_head, out_tail) = S::[<as_mut_simd_ $t s>](out);

                for ((x, l), r) in out_head
                    .iter_mut()
                    .zip(left_head)
                    .zip(right_head)
                {
                    *x = simd.[<$simd_method _ $t s>](*l, *r);
                }
                for ((x, l), r) in out_tail
                    .iter_mut()
                    .zip(left_tail.iter())
                    .zip(right_tail.iter())
                {
                    *x = l $op r
                }
            }
            )*
        }
    };
}

macro_rules! impl_simd_ops_inplace {
    ($t:ident, $($name:ident => $simd_method:ident => $op:tt),*) => {
        paste! {
            $(
            #[pulp::with_simd([<$name _inplace_ $t>] = pulp::Arch::new())]
            #[inline(always)]
            pub fn [<$name _inplace_impl_ $t>]<S: Simd>(simd: S, left: &mut [$t], right: &[$t]) {
                let (left_head, left_tail) = S::[<as_mut_simd_ $t s>](left);
                let (right_head, right_tail) = S::[<as_simd_ $t s>](right);

                for (l, r) in left_head.iter_mut().zip(right_head) {
                    *l = simd.[<$simd_method _ $t s>](*l, *r);
                }
                for (l, r) in left_tail.iter_mut().zip(right_tail) {
                    *l $op *r;
                }
            }
            )*
        }
    };
}

macro_rules! impl_simd_scalar_ops {
    ($t:ident, $($name:ident => $simd_method:ident => $op:tt),*) => {
        paste! {
            $(
            #[pulp::with_simd([<$name _scalar_to_out_ $t>] = pulp::Arch::new())]
            #[inline(always)]
            pub fn [<$name _scalar_to_out_impl_ $t>]<S: Simd>(simd: S, left: &[$t], right: $t, out: &mut [$t]) {
                let (left_head, left_tail) = S::[<as_simd_ $t s>](left);
                let rhs = simd.[<splat_ $t s>](right);
                let (out_head, out_tail) = S::[<as_mut_simd_ $t s>](out);

                for (l, o) in left_head.iter().zip(out_head) {
                    *o = simd.[<$simd_method _ $t s>](*l, rhs);
                }
                for (l, o) in left_tail.iter().zip(out_tail) {
                    *o = l $op right;
                }
            }
            )*
        }
    };
}

macro_rules! impl_simd_scalar_ops_inplace {
    ($t:ident, $($name:ident => $simd_method:ident => $op:tt),*) => {
        paste! {
            $(
            #[pulp::with_simd([<$name _scalar_inplace_ $t>] = pulp::Arch::new())]
            #[inline(always)]
            pub fn [<$name _scalar_inplace_impl_ $t>]<S: Simd>(simd: S, left: &mut [$t], right: $t) {
                let (left_head, left_tail) = S::[<as_mut_simd_ $t s>](left);
                let rhs = simd.[<splat_ $t s>](right);

                for l in left_head.iter_mut() {
                    *l = simd.[<$simd_method _ $t s>](*l, rhs);
                }
                for l in left_tail.iter_mut() {
                    *l $op right;
                }
            }
            )*
        }
    };
}

macro_rules! impl_simd_add_sub {
    ($($t:ident),*) => {
        $(
            impl_simd_ops!($t,
            add => add => +,
            sub => sub => -
            );
            impl_simd_scalar_ops!($t,
            add => add => +,
            sub => sub => -
            );
        )*
    };
}

macro_rules! impl_simd_mul {
    ($($t:ident),*) => {
        $(
            impl_simd_ops!($t,
            mul => mul => *
            );
            impl_simd_scalar_ops!($t,
            mul => mul => *
            );
        )*
    };
}

macro_rules! impl_simd_div {
    ($($t:ident),*) => {
        $(
            impl_simd_ops!($t,
            div => div => /
            );
            impl_simd_scalar_ops!($t,
            div => div => /
            );
        )*
    };
}

macro_rules! impl_simd_add_sub_inplace {
    ($($t:ident),*) => {
        $(
            impl_simd_ops_inplace!($t,
            add => add => +=,
            sub => sub => -=
            );
            impl_simd_scalar_ops_inplace!($t,
            add => add => +=,
            sub => sub => -=
            );
        )*
    };
}

macro_rules! impl_simd_mul_inplace {
    ($($t:ident),*) => {
        $(
            impl_simd_ops_inplace!($t,
            mul => mul => *=
            );
            impl_simd_scalar_ops_inplace!($t,
            mul => mul => *=
            );
        )*
    };
}

macro_rules! impl_simd_div_inplace {
    ($($t:ident),*) => {
        $(
            impl_simd_ops_inplace!($t,
            div => div => /=
            );
            impl_simd_scalar_ops_inplace!($t,
            div => div => /=
            );
        )*
    };
}

impl_simd_add_sub!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_simd_mul!(f32, f64, u16, u32, u64, i16, i32, i64);
impl_simd_div!(f32, f64);

impl_simd_add_sub_inplace!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_simd_mul_inplace!(f32, f64, u16, u32, u64, i16, i32, i64);
impl_simd_div_inplace!(f32, f64);

#[cfg(test)]
mod tests {
    use crate::vector_ops::Scalar;
    use crate::vector_ops::vector_add;
    use core::array;

    #[test]
    fn basic_add() {
        let x: [f64; 103] = array::from_fn(|x| x as f64 * 10.0);
        let y: [f64; 103] = array::from_fn(|x| x as f64 * 15.0);
        let expected: [f64; 103] = array::from_fn(|x| (x as f64 * 10.0) + (x as f64 * 15.0));

        let result = vector_add(&x, &y);

        assert_eq!(result, expected);
    }
    #[test]
    fn basic_scalar_mul() {
        let x: [f64; 103] = array::from_fn(|x| x as f64 * 10.3);
        let expected: [f64; 103] = array::from_fn(|i| x[i] * 5.0);

        let result = f64::vector_scalar_mul(&x, 5.0);

        assert_eq!(result, expected);
    }
}
