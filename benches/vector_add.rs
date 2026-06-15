use coalign::vector_ops::*;
use std::hint::black_box;
use std::ops::{Add, Div, Mul, Sub};

use coalign::VectorMapping;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::array;

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

fn bench_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_add");
    for size in [250, 1_000, 100_000, 10_000_000] {
        let a = vec![1.0f64; size];
        let b = vec![2.0f64; size];
        let mut out = vec![0.0f64; size];
        group.throughput(criterion::Throughput::Bytes((3 * 8 * size) as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bch, _| {
            bch.iter(|| {
                f64::vector_add_to_out(black_box(&a), black_box(&b), black_box(&mut out));
            });
        });
        let a = vec![NewFloat(1.0f64); size];
        let b = vec![NewFloat(2.0f64); size];
        let mut out = vec![NewFloat(0.0f64); size];
        group.bench_with_input(BenchmarkId::new("fallback", size), &size, |bch, _| {
            bch.iter(|| {
                NewFloat::vector_add_to_out(black_box(&a), black_box(&b), black_box(&mut out));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vector_add);
criterion_main!(benches);
