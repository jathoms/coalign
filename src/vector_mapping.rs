use crate::coalignment::realign_values_hash;
use crate::vector_ops::{Scalar, VectorAdd};
use crate::vector_ops::{VectorDiv, VectorMul, VectorOp, VectorSub};
use bytemuck::cast_slice;
use rustc_hash::FxHasher;
use std::hash::{BuildHasher, BuildHasherDefault, RandomState};
use std::sync::Arc;
use std::{
    cell::OnceCell,
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
    marker::PhantomData,
};
use xxhash_rust::xxh3::{Xxh3, xxh3_128};

use crate::errors::CoalignError;

pub trait ValueDomain: Scalar + Debug {}

impl<T: Scalar + Debug> ValueDomain for T {}

pub type Fingerprint = u128;

pub trait KeyDomain: Eq + Hash + Clone {
    type Hasher: BuildHasher + Default + Clone + Debug;
    fn build_pos(keys: &impl KeyContainer<Self>) -> HashMap<Self, usize, Self::Hasher> {
        keys.as_ref()
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, k)| (k, i))
            .collect()
    }
    fn fingerprint(keys: &impl KeyContainer<Self>) -> Fingerprint;
}

macro_rules! impl_keydomain_int {
    ($($t:ty),*) => {
        $(
            impl KeyDomain for $t {
                type Hasher = BuildHasherDefault<FxHasher>;
                fn build_pos(keys: &impl KeyContainer<$t>) -> HashMap<Self, usize, Self::Hasher> {
                    keys.as_ref()
                        .iter()
                        .enumerate()
                        .map(|(i, k)| (*k, i))
                        .collect()
                }
                fn fingerprint(keys: &impl KeyContainer<$t>) -> Fingerprint {
                    xxh3_128(cast_slice::<$t, u8>(keys.as_ref()))
                }
            }
        )*
    };
}

impl_keydomain_int!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

fn hash_byte_slices<'a>(items: impl Iterator<Item = &'a [u8]>, count: usize) -> Fingerprint {
    let mut h = Xxh3::new();
    h.write_usize(count); // count prefix
    for s in items {
        h.write_usize(s.len());
        h.write(s);
    }
    h.digest128()
}

impl KeyDomain for &[u8] {
    type Hasher = RandomState;
    fn fingerprint(keys: &impl KeyContainer<Self>) -> Fingerprint {
        let keys = keys.as_ref();
        hash_byte_slices(keys.iter().copied(), keys.len())
    }
}

impl KeyDomain for &str {
    type Hasher = RandomState;
    fn fingerprint(keys: &impl KeyContainer<Self>) -> Fingerprint {
        let keys = keys.as_ref();
        hash_byte_slices(keys.iter().map(|k| k.as_bytes()), keys.len())
    }
}

pub trait KeyContainer<K>: AsRef<[K]> + Clone {}
pub trait ValueContainer<V>: AsRef<[V]> + AsMut<[V]> + Clone {}

impl<T, C: AsRef<[T]> + Clone> KeyContainer<T> for C {}
impl<T, C: AsRef<[T]> + AsMut<[T]> + Clone> ValueContainer<T> for C {}
// impl<T: Clone> ValueContainer<T> for Vec<T> {}

#[derive(Default, Debug)]
struct OrderingContainer<K: KeyDomain, KC> {
    fingerprint: Fingerprint,
    labels: KC,
    pos: Arc<OnceCell<HashMap<K, usize, K::Hasher>>>,
}

impl<K: KeyDomain, KC: KeyContainer<K>> OrderingContainer<K, KC> {
    fn new(keys: KC) -> Self {
        let fingerprint = K::fingerprint(&keys);
        Self {
            fingerprint,
            labels: keys,
            pos: Default::default(),
        }
    }
}

#[derive(Default, Debug)]
pub struct VectorMapping<K: KeyDomain, V, KC, VC> {
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
            _marker: Default::default(),
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
            _marker: Default::default(),
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let idx = self
            .ordering
            .pos
            .get_or_init(|| K::build_pos(self.keys()))
            .get(key)?;
        self.values.as_ref().get(*idx)
    }

    pub fn keys(&self) -> &KC {
        &self.ordering.labels
    }
    pub fn values(&self) -> &VC {
        &self.values
    }

    fn aligned_to<VC2: ValueContainer<V>>(
        &self,
        target_order: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        let reordered_vals = realign_values_hash(
            self.ordering.labels.as_ref(),
            target_order.ordering.labels.as_ref(),
            self.values().as_ref(),
            self.ordering
                .pos
                .get_or_init(|| K::build_pos(&self.ordering.labels)),
        )?;
        Ok(VectorMapping::with_ordering(
            &target_order.ordering,
            reordered_vals,
        ))
    }
    fn op<O: VectorOp, VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        if self.ordering.fingerprint == rhs.ordering.fingerprint {
            Ok(VectorMapping::with_ordering(
                &self.ordering,
                O::apply(&self.values.as_ref(), &rhs.values.as_ref()),
            ))
        } else {
            let rhs_aligned_to_lhs = rhs.aligned_to(self)?;
            let new_vals = O::apply(&self.values.as_ref(), &rhs_aligned_to_lhs.values.as_ref());
            Ok(VectorMapping::with_ordering(&self.ordering, new_vals))
        }
    }
    fn op_inplace<O: VectorOp, VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        if self.ordering.fingerprint == rhs.ordering.fingerprint {
            O::apply_inplace(self.values.as_mut(), rhs.values.as_ref());
        } else {
            let rhs_aligned_to_lhs = rhs.aligned_to(self)?;
            O::apply_inplace(self.values.as_mut(), rhs_aligned_to_lhs.values.as_ref());
        }
        Ok(())
    }

    pub fn add<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorAdd, VC2>(&rhs)
    }
    pub fn sub<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorSub, VC2>(&rhs)
    }
    pub fn mul<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorMul, VC2>(&rhs)
    }
    pub fn div<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorDiv, VC2>(&rhs)
    }
    pub fn add_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorAdd, VC2>(&rhs)
    }
    pub fn sub_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorSub, VC2>(&rhs)
    }
    pub fn mul_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorMul, VC2>(&rhs)
    }
    pub fn div_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorDiv, VC2>(&rhs)
    }

    #[must_use]
    pub fn from_map_unsorted(map: &HashMap<K, V>) -> VectorMapping<K, V, Vec<K>, Vec<V>> {
        let keys = map.keys().cloned().collect::<Vec<_>>();
        let values = map.values().copied().collect::<Vec<_>>();

        VectorMapping::new(keys, values)
    }
}
impl<K: KeyDomain + Ord, V: ValueDomain> VectorMapping<K, V, Vec<K>, Vec<V>> {
    #[must_use]
    pub fn from_map_sorted(map: &HashMap<K, V>) -> Self {
        let mut k_v_sorted = map
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect::<Vec<(K, V)>>();
        k_v_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let (sorted_keys, sorted_values) = k_v_sorted.into_iter().unzip();

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
    fn align_mapping() {
        let a = VectorMapping::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let b = VectorMapping::new(vec![2, 0, 1], vec![10.0, 20.0, 30.0]);
        let c = b.aligned_to(&a).expect("Couldn't align mappings");
        dbg!(&c);
        assert_eq!(
            <Vec<f64> as AsRef<Vec<f64>>>::as_ref(&c.values()),
            &[20.0, 30.0, 10.0]
        );
    }
}
