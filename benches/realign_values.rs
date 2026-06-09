use coalign::vector_mapping::KeyDomain;
use std::{collections::HashMap, hint::black_box};

use coalign::coalignment::realign_values_same_len;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn bench_realign_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("realign_values");

    for size in [64usize, 256, 1_024, 4_096] {
        let lhs = (0..size as u32).collect::<Vec<_>>();
        let rhs = lhs.iter().rev().copied().collect::<Vec<_>>();
        let misaligned_values = lhs.iter().map(|key| *key as f64).collect::<Vec<_>>();
        let pos_mapping_lhs_precomputed = lhs
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, key)| (key, idx))
            .collect::<HashMap<_, _, <u32 as KeyDomain>::Hasher>>();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("hash", size), &size, |bch, _| {
            bch.iter(|| {
                let pos_mapping_lhs = black_box(
                    lhs.iter()
                        .copied()
                        .enumerate()
                        .map(|(idx, key)| (key, idx))
                        .collect::<HashMap<_, _, <u32 as KeyDomain>::Hasher>>(),
                );
                black_box(
                    realign_values_same_len(
                        black_box(&lhs),
                        black_box(&rhs),
                        black_box(&misaligned_values),
                        black_box(&pos_mapping_lhs),
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_with_input(
            BenchmarkId::new("hash_precomputed", size),
            &size,
            |bch, _| {
                bch.iter(|| {
                    black_box(
                        realign_values_same_len(
                            black_box(&lhs),
                            black_box(&rhs),
                            black_box(&misaligned_values),
                            black_box(&pos_mapping_lhs_precomputed),
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_realign_values);
criterion_main!(benches);
