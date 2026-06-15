use coalign::{VectorMapping, vector_ops::*};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::ops::{Add, Div, Mul, Sub};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::seq::SliceRandom;

const SIZES: [usize; 4] = [250, 1_000, 100_000, 500_000];

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
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
impl From<f64> for NewFloat {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl Scalar for NewFloat {}

type VecMap<V> = VectorMapping<i32, V, Vec<i32>, Vec<V>>;

#[derive(Clone)]
struct MixedOpsInput<V> {
    k1: Vec<i32>,
    v1: Vec<V>,
    k2: Vec<i32>,
    v2: Vec<V>,
}

fn build_mixed_ops_input<V>(size: usize) -> MixedOpsInput<V>
where
    V: Scalar + From<f64> + std::fmt::Debug,
{
    let k1 = (0..size).map(|i| (i * 32) as i32 - 5).collect::<Vec<_>>();
    let v1 = (0..size)
        .map(|i| V::from((((i << 5) % (i + 1) * 1000) / (i + 1)) as f64))
        .collect::<Vec<_>>();

    let mut k2 = k1.clone();
    let mut v2 = v1.clone();

    let mut rng = StdRng::seed_from_u64(41);
    k2.shuffle(&mut rng);
    let mut rng = StdRng::seed_from_u64(41);
    v2.shuffle(&mut rng);

    MixedOpsInput { k1, v1, k2, v2 }
}

fn build_vector_maps<V>(input: &MixedOpsInput<V>) -> (VecMap<V>, VecMap<V>)
where
    V: Scalar + std::fmt::Debug,
{
    let vec_map1 = VectorMapping::new(input.k1.clone(), input.v1.clone());
    let vec_map2 = VectorMapping::new(input.k2.clone(), input.v2.clone());

    if let Some(key) = vec_map2.keys().first() {
        let _ = vec_map2.get(key);
    }

    (vec_map1, vec_map2)
}

fn perform_mixed_ops<V>(vec_map1: &VecMap<V>, vec_map2: &VecMap<V>) -> VecMap<V>
where
    V: Scalar + From<f64> + std::fmt::Debug,
{
    vec_map1
        .mul(vec_map2)
        .unwrap()
        .add_scalar(V::from(17.0))
        .div_scalar(V::from(13.0))
        .mul(vec_map2)
        .unwrap()
}

fn perform_mixed_ops_inplace<V>(vec_map1: &mut VecMap<V>, vec_map2: &VecMap<V>)
where
    V: Scalar + From<f64> + std::fmt::Debug,
{
    vec_map1.mul_inplace(vec_map2).unwrap();
    vec_map1.add_scalar_inplace(V::from(17.0));
    vec_map1.div_scalar_inplace(V::from(13.0));
    vec_map1.mul_inplace(vec_map2).unwrap();
}

macro_rules! bench_mixed_size {
    ($group:expr, $size:expr) => {{
        let size = $size;
        $group.throughput(criterion::Throughput::Elements(size as u64));

        let simd_input = build_mixed_ops_input::<f64>(size);
        let (simd_vec_map1, simd_vec_map2) = build_vector_maps(&simd_input);
        $group.bench_with_input(BenchmarkId::new("simd", size), &size, |bch, _| {
            bch.iter(|| {
                let result = black_box(perform_mixed_ops(
                    black_box(&simd_vec_map1),
                    black_box(&simd_vec_map2),
                ));
                black_box(result)
            });
        });

        let fallback_input = build_mixed_ops_input::<NewFloat>(size);
        let (fallback_vec_map1, fallback_vec_map2) = build_vector_maps(&fallback_input);
        $group.bench_with_input(BenchmarkId::new("fallback", size), &size, |bch, _| {
            bch.iter(|| {
                let result = black_box(perform_mixed_ops(
                    black_box(&fallback_vec_map1),
                    black_box(&fallback_vec_map2),
                ));
                black_box(result)
            });
        });

        let simd_inplace_input = build_mixed_ops_input::<f64>(size);
        $group.bench_with_input(BenchmarkId::new("simd_inplace", size), &size, |bch, _| {
            bch.iter_batched(
                || build_vector_maps(&simd_inplace_input),
                |(mut vec_map1, vec_map2)| {
                    perform_mixed_ops_inplace(black_box(&mut vec_map1), black_box(&vec_map2));
                    black_box(vec_map1)
                },
                BatchSize::LargeInput,
            );
        });

        let fallback_inplace_input = build_mixed_ops_input::<NewFloat>(size);
        $group.bench_with_input(
            BenchmarkId::new("fallback_inplace", size),
            &size,
            |bch, _| {
                bch.iter_batched(
                    || build_vector_maps(&fallback_inplace_input),
                    |(mut vec_map1, vec_map2)| {
                        perform_mixed_ops_inplace(black_box(&mut vec_map1), black_box(&vec_map2));
                        black_box(vec_map1)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }};
}

fn bench_mixed_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_ops");
    bench_mixed_size!(group, SIZES[0]);
    bench_mixed_size!(group, SIZES[1]);
    bench_mixed_size!(group, SIZES[2]);
    bench_mixed_size!(group, SIZES[3]);
    group.finish();
}

criterion_group!(benches, bench_mixed_ops);
criterion_main!(benches);
