use coalign::VectorMapping;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::array;

#[test]
fn check_adding_mappings() {
    let k1 = array::from_fn::<i32, 100, _>(|i| (i * 32) as i32 - 5);
    let v1 = array::from_fn::<_, 100, _>(|i| (((i << 5) % (i + 1) * 1000) / (i + 1)) as f64);

    let mut k2 = k1.clone();
    let mut v2 = v1.clone();

    let mut rng = StdRng::seed_from_u64(67);
    k2.shuffle(&mut rng);
    let mut rng = StdRng::seed_from_u64(67);
    v2.shuffle(&mut rng);

    let vec_map1 = VectorMapping::new(k1, v1);
    let vec_map2 = VectorMapping::new(k2, v2);

    let result = vec_map1
        .add(&vec_map2)
        .expect("Failed to add vector mappings");

    dbg!(&vec_map1, &vec_map2, &result);

    assert_eq!(
        result.values(),
        &vec_map1
            .values()
            .iter()
            .map(|v| v * 2.0)
            .collect::<Vec<_>>()
    )
}
