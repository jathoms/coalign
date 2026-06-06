use std::collections::HashMap;

use crate::{
    errors::CoalignError,
    vector_mapping::{KeyDomain, ValueDomain},
};

pub fn realign_values_linear<K, V>(
    lhs: &[K],
    rhs: &[K],
    misaligned_values: &[V],
) -> Result<Vec<V>, CoalignError>
where
    K: KeyDomain,
    V: ValueDomain,
{
    // misaligned_values corresponds to lhs' values,
    // we want to align them to rhs' keys
    let len = rhs.len();
    assert_eq!(lhs.len(), len);

    let mut reordered_vals = vec![V::default(); len];
    let mut used = vec![false; len];
    let mut num_filled = 0;

    for (i, k) in rhs.iter().enumerate() {
        let pos = lhs
            .iter()
            .enumerate()
            .position(|(_, e)| e == k)
            .ok_or(CoalignError::IncompatibleKeys)?;
        if used[pos] {
            return Err(CoalignError::IncompatibleKeys);
        }
        used[pos] = true;
        num_filled += 1;
        reordered_vals[i] = misaligned_values[pos]
    }
    if num_filled < len {
        return Err(CoalignError::IncompatibleKeys);
    };
    Ok(reordered_vals)
}

pub fn realign_values_hash<K, V>(
    lhs: &[K],
    rhs: &[K],
    misaligned_values: &[V],
    pos_mapping_lhs: &HashMap<K, usize, K::Hasher>,
) -> Result<Vec<V>, CoalignError>
where
    K: KeyDomain,
    V: ValueDomain,
{
    // misaligned_values corresponds to lhs' values,
    // we want to align them to rhs' keys
    let len = rhs.len();
    assert_eq!(lhs.len(), len);

    let mut reordered_vals = vec![V::default(); len];
    let mut used = vec![false; len];
    let mut num_filled = 0;

    for (i, k) in rhs.iter().enumerate() {
        let pos = *pos_mapping_lhs
            .get(k)
            .ok_or(CoalignError::IncompatibleKeys)?;
        if used[pos] {
            return Err(CoalignError::IncompatibleKeys);
        }
        used[pos] = true;
        num_filled += 1;
        reordered_vals[i] = misaligned_values[pos]
    }
    if num_filled < len {
        return Err(CoalignError::IncompatibleKeys);
    };
    Ok(reordered_vals)
}
