use pulp::Simd;

#[pulp::with_simd(add_to_out = pulp::Arch::new())]
#[inline(always)]
pub fn add_to_out_impl<S: Simd>(simd: S, left: &[f64], right: &[f64], out: &mut [f64]) {
    let (left_head, left_tail) = S::as_simd_f64s(left);
    let (right_head, right_tail) = S::as_simd_f64s(right);
    let (out_head, out_tail) = S::as_mut_simd_f64s(out);

    for ((x, l), r) in out_head
        .iter_mut()
        .zip(left_head.iter())
        .zip(right_head.iter())
    {
        *x = simd.add_f64s(*l, *r);
    }
    for ((x, l), r) in out_tail
        .iter_mut()
        .zip(left_tail.iter())
        .zip(right_tail.iter())
    {
        *x = l + r
    }
}

#[pulp::with_simd(add_inplace = pulp::Arch::new())]
#[inline(always)]
pub fn add_inplace_impl<S: Simd>(simd: S, left: &mut [f64], right: &[f64]) {
    let (left_head, left_tail) = S::as_mut_simd_f64s(left);
    let (right_head, right_tail) = S::as_simd_f64s(right);

    for (l, r) in left_head.iter_mut().zip(right_head) {
        *l = simd.add_f64s(*l, *r);
    }
    for (l, r) in left_tail.iter_mut().zip(right_tail) {
        *l += *r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array;

    #[test]
    fn basic_add() {
        let x: [f64; 100] = array::from_fn(|x| x as f64 * 10.0);
        let y: [f64; 100] = array::from_fn(|x| x as f64 * 15.0);
        let expected: [f64; 100] = array::from_fn(|x| (x as f64 * 10.0) + (x as f64 * 15.0));

        let mut result: [f64; 100] = array::from_fn(|_| 0.0);

        add_to_out(&x, &y, &mut result);

        assert_eq!(result, expected);
    }
}
