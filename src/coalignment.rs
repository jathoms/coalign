use std::collections::HashMap;

use crate::{
    errors::CoalignError,
    vector_mapping::{KeyDomain, ValueDomain},
};

/// `misaligned_values` corresponds to lhs' values,
/// this function aligns them to rhs' keys
pub fn realign_values_linear<K, V>(
    lhs: &[K],
    rhs: &[K],
    misaligned_values: &[V],
) -> Result<Vec<V>, CoalignError>
where
    K: KeyDomain,
    V: ValueDomain,
{
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
        reordered_vals[i] = misaligned_values[pos];
    }
    if num_filled < len {
        return Err(CoalignError::IncompatibleKeys);
    }
    Ok(reordered_vals)
}

/// `misaligned_values` corresponds to lhs' values,
/// this function aligns them to rhs' keys
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
        reordered_vals[i] = misaligned_values[pos];
    }
    if num_filled < len {
        return Err(CoalignError::IncompatibleKeys);
    }
    Ok(reordered_vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_realign() {
        let k1 = ["a", "b", "c"];
        let k2 = ["c", "b", "a"];
        let vals = [3.0, 2.0, 1.0];
        let pos_mapping_lhs = k1.iter().enumerate().map(|(i, k)| (*k, i)).collect();

        let realigned = realign_values_hash(&k1, &k2, &vals, &pos_mapping_lhs)
            .expect("Failed to realign values");

        assert_eq!(realigned, [1.0, 2.0, 3.0]);
    }
}
