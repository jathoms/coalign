pub mod errors;
pub mod simd;
pub mod vector_mapping;
pub mod vector_ops;
pub use vector_mapping::VectorMapping;
pub mod coalignment;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn add_same_ordering() {
        let a = VectorMapping::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let b = VectorMapping::new(vec![0, 1, 2], vec![10.0, 20.0, 30.0]);
        let c = a.add(&b).expect("Couldn't add nums");
        assert_eq!(
            <Vec<f64> as AsRef<Vec<f64>>>::as_ref(&c.values()),
            &[11.0, 22.0, 33.0]
        );
    }
    #[test]
    fn add_diff_ordering() {
        let a = VectorMapping::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let b = VectorMapping::new(vec![2, 0, 1], vec![10.0, 20.0, 30.0]);
        let c = a.add(&b).expect("Couldn't add nums");
        dbg!(&c);
        let d = c.add(&a).expect("Couldn't add nums");
        dbg!(&d);
        assert_eq!(
            <Vec<f64> as AsRef<Vec<f64>>>::as_ref(&d.values()),
            &[22.0, 34.0, 16.0]
        );
    }
    #[test]
    fn from_map() {
        let tups = [(1, 10.0), (2, 11.0), (3, 12.0)];
        let map: HashMap<u32, f64> = tups.into();
        let vmap: VectorMapping<u32, f64, Vec<u32>, Vec<f64>> = map.into();
        dbg!(vmap);
    }
}
