use coalign::vector_ops::*;
use std::hint::black_box;
use std::ops::{Add, Div, Mul, Sub};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
struct NewFloat(f64);
impl Add for NewFloat {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        NewFloat(self.0 + r.0)
    }
}
impl Sub for NewFloat {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        NewFloat(self.0 - r.0)
    }
}
impl Mul for NewFloat {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        NewFloat(self.0 * r.0)
    }
}
impl Div for NewFloat {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        NewFloat(self.0 / r.0)
    }
}
impl Scalar for NewFloat {}

#[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
struct NewInt(u32);
impl Add for NewInt {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        NewInt(self.0 + r.0)
    }
}
impl Sub for NewInt {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        NewInt(self.0 - r.0)
    }
}
impl Mul for NewInt {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        NewInt(self.0 * r.0)
    }
}
impl Div for NewInt {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        NewInt(self.0 / r.0)
    }
}
impl Scalar for NewInt {}

fn bench_scalar_add_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("int_scalar_add");
    for size in [250, 1_000, 100_000, 10_000_000] {
        let a = vec![1.0f64; size];
        let b_scalar = 2.0;
        let mut out = vec![0.0f64; size];
        group.throughput(criterion::Throughput::Bytes((2 * 8 * size) as u64));
        group.bench_with_input(BenchmarkId::new("float_simd", size), &size, |bch, _| {
            bch.iter(|| {
                f64::vector_scalar_add_to_out(
                    black_box(&a),
                    black_box(b_scalar),
                    black_box(&mut out),
                );
            });
        });
        let a = vec![NewFloat(3.0); size];
        let b = NewFloat(4.0);
        let mut out = vec![NewFloat(0.0); size];
        group.bench_with_input(BenchmarkId::new("float_fallback", size), &size, |bch, _| {
            bch.iter(|| {
                NewFloat::vector_scalar_add_to_out(
                    black_box(&a),
                    black_box(b),
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn bench_scalar_add_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_add");
    for size in [250, 1_000, 100_000, 10_000_000] {
        let a = vec![3; size];
        let b_scalar = 2;
        let mut out = vec![0; size];
        group.throughput(criterion::Throughput::Bytes((2 * 4 * size) as u64));
        group.bench_with_input(BenchmarkId::new("int_simd", size), &size, |bch, _| {
            bch.iter(|| {
                u32::vector_scalar_add_to_out(
                    black_box(&a),
                    black_box(b_scalar),
                    black_box(&mut out),
                );
            });
        });
        let a = vec![NewInt(1); size];
        let b = NewInt(2);
        let mut out = vec![NewInt(0); size];
        group.bench_with_input(BenchmarkId::new("int_fallback", size), &size, |bch, _| {
            bch.iter(|| {
                NewInt::vector_scalar_add_to_out(black_box(&a), black_box(b), black_box(&mut out));
            });
        });
    }
    group.finish();
}
criterion_group!(benches, bench_scalar_add_u32, bench_scalar_add_f64);
criterion_main!(benches);
