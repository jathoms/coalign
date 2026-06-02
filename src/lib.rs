mod errors;
mod vector_mapping;
mod vector_ops;
use bytemuck::cast_slice;
use std::{
    cell::OnceCell,
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};
use xxhash_rust::xxh3::{Xxh3, xxh3_128};

use crate::errors::CoalignError;

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
}
impl<
    T: Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Sized
        + Copy
        + Default
        + PartialOrd,
> Scalar for T
{
}

pub trait ValueDomain: Scalar + Debug {}

impl<T: Scalar + Debug> ValueDomain for T {}

pub type Fingerprint = u128;

pub trait KeyDomain: Default + Eq + Hash + Clone {
    fn build_pos(keys: &impl KeyContainer<Self>) -> HashMap<Self, usize> {
        keys.as_ref()
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, k)| (k, i))
            .collect()
    }
    fn fingerprint(keys: &impl KeyContainer<Self>) -> Fingerprint;
}

impl KeyDomain for u32 {
    fn fingerprint(keys: &impl KeyContainer<u32>) -> Fingerprint {
        xxh3_128(cast_slice::<u32, u8>(keys.as_ref()))
    }
}

impl KeyDomain for String {
    fn fingerprint(keys: &impl KeyContainer<String>) -> Fingerprint {
        let mut h = Xxh3::new();
        h.write_usize(keys.as_ref().len()); // count prefix
        for s in keys.as_ref().iter() {
            h.write_usize(s.len());
            h.write(s.as_bytes());
        }
        h.digest128()
    }
}

pub trait KeyContainer<K>: AsRef<[K]> + FromIterator<K> + Default + Clone {}
pub trait ValueContainer<V>: AsRef<[V]> + FromIterator<V> + Default + Clone {}

impl<T: Clone> KeyContainer<T> for Vec<T> {}
impl<T: Clone> ValueContainer<T> for Vec<T> {}

#[derive(Default, Debug)]
struct OrderingContainer<K: KeyDomain, KC: KeyContainer<K>> {
    fingerprint: Fingerprint,
    labels: KC,
    pos: OnceCell<HashMap<K, usize>>,
}

impl<K: KeyDomain, KC: KeyContainer<K>> OrderingContainer<K, KC> {
    fn new(keys: KC) -> Self {
        let fingerprint = K::fingerprint(&keys);
        Self {
            fingerprint,
            labels: keys,
            pos: OnceCell::new(),
        }
    }
}

#[derive(Default, Debug)]
pub struct VectorMapping<
    K: KeyDomain,
    V: ValueDomain,
    KC: KeyContainer<K>,
    VC: AsRef<[V]> + Default,
> {
    ordering: OrderingContainer<K, KC>,
    values: VC,
    _marker: PhantomData<V>,
}

impl<K: KeyDomain, V: ValueDomain, KC: KeyContainer<K>, VC: ValueContainer<V>>
    VectorMapping<K, V, KC, VC>
{
    pub fn new(keys: KC, values: VC) -> Self {
        assert_eq!(keys.as_ref().len(), values.as_ref().len());
        Self {
            ordering: OrderingContainer::new(keys),
            values,
            ..Default::default()
        }
    }

    fn with_ordering(ordering: &OrderingContainer<K, KC>, values: VC) -> Self {
        Self {
            ordering: OrderingContainer {
                fingerprint: ordering.fingerprint,
                labels: ordering.labels.clone(),
                pos: ordering.pos.clone(),
            },
            values,
            ..Default::default()
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let idx = self
            .ordering
            .pos
            .get_or_init(|| K::build_pos(self.keys()))
            .get(key)?;
        Some(self.values.as_ref().get(*idx)?)
    }

    pub fn keys(&self) -> &KC {
        &self.ordering.labels
    }
    pub fn values(&self) -> &VC {
        &self.values
    }

    fn aligned_to(
        &self,
        ordering: &OrderingContainer<K, KC>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        assert_eq!(
            ordering.labels.as_ref().len(),
            self.ordering.labels.as_ref().len()
        );
        let len = ordering.labels.as_ref().len();

        let target_pos = ordering.pos.get_or_init(|| K::build_pos(&ordering.labels));

        let mut reordered_vals = vec![V::default(); len];
        let mut filled = vec![false; len];

        for (k, &pos) in target_pos.iter() {
            if filled[pos] == true {
                return Err(CoalignError::IncompatibleKeys);
            }
            filled[pos] = true;
            reordered_vals[pos] = self.get(k).cloned().ok_or(CoalignError::KeyError)?;
        }
        if filled.iter().any(|x| !x) {
            return Err(CoalignError::IncompatibleKeys);
        }
        Ok(VectorMapping::with_ordering(&ordering, reordered_vals))
    }
    pub fn add(&self, rhs: &Self) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        if self.ordering.fingerprint == rhs.ordering.fingerprint {
            return Ok(VectorMapping::with_ordering(
                &self.ordering,
                vector_ops::vector_add(&self.values, &rhs.values),
            ));
        } else {
            let rhs_aligned_to_lhs = rhs.aligned_to(&self.ordering)?;
            return Ok(self.add_aligned(&rhs_aligned_to_lhs));
        }
    }
    fn add_aligned<VC2>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> VectorMapping<K, V, KC, Vec<V>>
    where
        VC2: ValueContainer<V>,
    {
        let new_vals = vector_ops::vector_add(&self.values, &rhs.values);
        return VectorMapping::with_ordering(&self.ordering, new_vals);
    }
    pub fn from_map_unsorted(map: &HashMap<K, V>) -> VectorMapping<K, V, Vec<K>, Vec<V>> {
        let keys = map.keys().cloned().collect::<Vec<_>>();
        let values = map.values().cloned().collect::<Vec<_>>();

        VectorMapping::new(keys, values)
    }
}
impl<K: KeyDomain + Ord, V: ValueDomain> VectorMapping<K, V, Vec<K>, Vec<V>> {
    pub fn from_map_sorted(map: &HashMap<K, V>) -> Self {
        let mut k_v_sorted = map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<(K, V)>>();
        k_v_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let (sorted_keys, sorted_values) = k_v_sorted.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

        VectorMapping::new(sorted_keys, sorted_values)
    }
}

impl<K: KeyDomain + Ord, V: ValueDomain> From<HashMap<K, V>>
    for VectorMapping<K, V, Vec<K>, Vec<V>>
{
    fn from(value: HashMap<K, V>) -> Self {
        VectorMapping::from_map_sorted(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn from_into_map() {
        let map_into = [(1, 10.0), (2, 11.0), (3, 12.0)];
        let map: HashMap<u32, f64> = map_into.clone().into();
        let vmap: VectorMapping<u32, f64, Vec<u32>, Vec<f64>> = map.into();
        dbg!(vmap);
    }
}
