use crate::coalignment::realign_values_same_len;
use crate::vector_ops::{Scalar, VectorAdd};
use crate::vector_ops::{VectorDiv, VectorMul, VectorOp, VectorSub};
use bytemuck::cast_slice;
use rustc_hash::FxHasher;
use std::borrow::Borrow;
use std::hash::{BuildHasher, BuildHasherDefault, RandomState};
use std::sync::{Arc, OnceLock};
use std::{
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

impl KeyDomain for Arc<str> {
    type Hasher = RandomState;
    fn fingerprint(keys: &impl KeyContainer<Self>) -> Fingerprint {
        let keys = keys.as_ref();
        hash_byte_slices(keys.iter().map(|k| k.as_bytes()), keys.len())
    }
}

pub trait KeyContainer<K>: AsRef<[K]> + Clone {}
pub trait ValueContainer<V>: AsRef<[V]> + Clone {}
pub trait MutValueContainer<V>: AsRef<[V]> + AsMut<[V]> + Clone {}

impl<T, C: AsRef<[T]> + Clone> KeyContainer<T> for C {}
impl<T, C: AsRef<[T]> + Clone> ValueContainer<T> for C {}
impl<T, C: AsRef<[T]> + AsMut<[T]> + Clone> MutValueContainer<T> for C {}

#[derive(Default, Debug, Clone)]
pub struct OrderingContainer<K: KeyDomain, KC> {
    fingerprint: Fingerprint,
    labels: KC,
    pos: Arc<OnceLock<HashMap<K, usize, K::Hasher>>>,
}

impl<K: KeyDomain, KC: KeyContainer<K>> OrderingContainer<K, KC> {
    pub fn new(keys: KC) -> Self {
        let fingerprint = K::fingerprint(&keys);
        Self {
            fingerprint,
            labels: keys,
            pos: Default::default(),
        }
    }
}

#[derive(Default, Debug, Clone)]
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
        assert_eq!(
            ordering.labels.as_ref().len(),
            values.as_ref().len(),
            "Length of ordering and values do not match in `VectorMapping::with_ordering`"
        );
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

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let idx = self
            .ordering
            .pos
            .get_or_init(|| K::build_pos(self.keys()))
            .get(key.borrow())?;
        self.values.as_ref().get(*idx)
    }

    pub fn get_idx(&self, idx: usize) -> Option<&V> {
        self.values.as_ref().get(idx)
    }

    pub fn keys(&self) -> &KC {
        &self.ordering.labels
    }

    pub fn values(&self) -> &VC {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.ordering.labels.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn aligned_to<VC2: ValueContainer<V>>(
        &self,
        target_order: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        let reordered_vals = realign_values_same_len(
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
    pub fn into_parts(self) -> (OrderingContainer<K, KC>, VC) {
        (self.ordering, self.values)
    }
    pub fn from_parts<VC2>(
        ordering: OrderingContainer<K, KC>,
        values: VC2,
    ) -> VectorMapping<K, V, KC, VC2>
    where
        KC: KeyContainer<K>,
        VC2: ValueContainer<V>,
    {
        assert_eq!(ordering.labels.as_ref().len(), values.as_ref().len());
        VectorMapping {
            ordering,
            values,
            _marker: Default::default(),
        }
    }

    fn op<O: VectorOp, VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        if self.ordering.fingerprint == rhs.ordering.fingerprint {
            return Ok(VectorMapping::with_ordering(
                &self.ordering,
                O::apply(self.values.as_ref(), rhs.values.as_ref()),
            ));
        };
        if self.len() == rhs.len() {
            let rhs_aligned_to_lhs = rhs.aligned_to(self)?;
            let new_vals = O::apply(self.values.as_ref(), rhs_aligned_to_lhs.values.as_ref());
            return Ok(VectorMapping::with_ordering(&self.ordering, new_vals));
        };
        // 2 paths when adding a A and B where A and B are `VectorMapping`s and
        // B's keys are a subset of A's or vice-versa
        // 1. create a new vector of <identity> and fill in the values of the smaller vector in the
        //    correct places, then perform a standard vector op between the values and the constructed
        //    vector, or
        // 2. copy the larger vector into a new buffer and edit the specific elements 1-by-1.
        //
        // for now, only approach 2 is implemented
        if self.len() > rhs.len() {
            Self::op_large_small::<O, VC, VC2>(self, rhs)
        } else {
            Self::op_large_small::<O, VC2, VC>(rhs, self)
        }
    }
    fn op_large_small<O: VectorOp, L: ValueContainer<V>, S: ValueContainer<V>>(
        large: &VectorMapping<K, V, KC, L>,
        small: &VectorMapping<K, V, KC, S>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        let large_pos = large
            .ordering
            .pos
            .get_or_init(|| K::build_pos(&large.ordering.labels));
        let mut buf = large.values().as_ref().to_owned();

        for (small_k, small_v) in small.keys().as_ref().iter().zip(small.values().as_ref()) {
            let pos = large_pos
                .get(small_k)
                .ok_or(CoalignError::IncompatibleKeys)?;
            buf[*pos] = O::apply_scalar_scalar(buf[*pos], *small_v)
        }
        Ok(VectorMapping::with_ordering(&large.ordering, buf))
    }

    pub fn op_scalar<O: VectorOp>(&self, rhs: V) -> VectorMapping<K, V, KC, Vec<V>> {
        VectorMapping::with_ordering(&self.ordering, O::apply_scalar(self.values.as_ref(), rhs))
    }

    pub fn add<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorAdd, VC2>(rhs)
    }

    pub fn sub<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorSub, VC2>(rhs)
    }

    pub fn mul<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorMul, VC2>(rhs)
    }

    pub fn div<VC2: ValueContainer<V>>(
        &self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<VectorMapping<K, V, KC, Vec<V>>, CoalignError> {
        self.op::<VectorDiv, VC2>(rhs)
    }

    pub fn add_scalar(&self, rhs: V) -> VectorMapping<K, V, KC, Vec<V>> {
        self.op_scalar::<VectorAdd>(rhs)
    }

    pub fn sub_scalar(&self, rhs: V) -> VectorMapping<K, V, KC, Vec<V>> {
        self.op_scalar::<VectorSub>(rhs)
    }

    pub fn mul_scalar(&self, rhs: V) -> VectorMapping<K, V, KC, Vec<V>> {
        self.op_scalar::<VectorMul>(rhs)
    }

    pub fn div_scalar(&self, rhs: V) -> VectorMapping<K, V, KC, Vec<V>> {
        self.op_scalar::<VectorDiv>(rhs)
    }

    #[must_use]
    pub fn from_map_unsorted(map: &HashMap<K, V>) -> VectorMapping<K, V, Vec<K>, Vec<V>> {
        let keys = map.keys().cloned().collect::<Vec<_>>();
        let values = map.values().copied().collect::<Vec<_>>();

        VectorMapping::new(keys, values)
    }
}

impl<K: KeyDomain, V: ValueDomain, KC: KeyContainer<K>, VC: MutValueContainer<V>>
    VectorMapping<K, V, KC, VC>
{
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
    fn op_scalar_inplace<O: VectorOp>(&mut self, rhs: V) {
        O::apply_scalar_inplace(self.values.as_mut(), rhs);
    }
    pub fn add_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorAdd, VC2>(rhs)
    }

    pub fn sub_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorSub, VC2>(rhs)
    }

    pub fn mul_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorMul, VC2>(rhs)
    }

    pub fn div_inplace<VC2: ValueContainer<V>>(
        &mut self,
        rhs: &VectorMapping<K, V, KC, VC2>,
    ) -> Result<(), CoalignError> {
        self.op_inplace::<VectorDiv, VC2>(rhs)
    }

    pub fn add_scalar_inplace(&mut self, rhs: V) {
        self.op_scalar_inplace::<VectorAdd>(rhs);
    }

    pub fn sub_scalar_inplace(&mut self, rhs: V) {
        self.op_scalar_inplace::<VectorSub>(rhs);
    }

    pub fn mul_scalar_inplace(&mut self, rhs: V) {
        self.op_scalar_inplace::<VectorMul>(rhs);
    }

    pub fn div_scalar_inplace(&mut self, rhs: V) {
        self.op_scalar_inplace::<VectorDiv>(rhs);
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
    #[test]
    fn add_subset_mapping() {
        let a = VectorMapping::new(vec!["a", "b", "c"], vec![1.0, 2.0, 3.0]);
        let b = VectorMapping::new(vec!["b"], vec![20.0]);
        let c = a.add(&b).expect("Couldn't add mappings");
        let d = b.add(&a).expect("Couldn't add mappings");
        assert_eq!(c.keys(), d.keys());
        assert_eq!(c.values(), d.values());
        dbg!(&c);
        assert_eq!(
            <Vec<f64> as AsRef<Vec<f64>>>::as_ref(&c.values()),
            &[1.0, 22.0, 3.0]
        );
    }
}
