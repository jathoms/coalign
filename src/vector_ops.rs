use crate::Scalar;
use crate::ValueContainer;
pub fn vector_add<V, VC, VC2>(left: &VC, right: &VC2) -> Vec<V>
where
    V: Scalar,
    VC: ValueContainer<V>,
    VC2: ValueContainer<V>,
{
    // TODO: SIMD/vectorised dispatch, for now just basic element-wise add
    let len = left.as_ref().len();
    assert_eq!(len, right.as_ref().len());
    left.as_ref()
        .iter()
        .zip(right.as_ref().iter())
        .map(|(e1, e2)| *e1 + *e2)
        .collect()
}

pub fn vector_sub<V, VC, VC2>(left: &VC, right: &VC2) -> Vec<V>
where
    V: Scalar,
    VC: ValueContainer<V>,
    VC2: ValueContainer<V>,
{
    // TODO: SIMD/vectorised dispatch
    let len = left.as_ref().len();
    assert_eq!(len, right.as_ref().len());
    left.as_ref()
        .iter()
        .zip(right.as_ref().iter())
        .map(|(e1, e2)| *e1 - *e2)
        .collect()
}

pub fn vector_mul<V, VC>(left: &VC, right: &VC) -> Vec<V>
where
    V: Scalar,
    VC: ValueContainer<V>,
{
    // TODO: SIMD/vectorised dispatch
    let len = left.as_ref().len();
    assert_eq!(len, right.as_ref().len());
    left.as_ref()
        .iter()
        .zip(right.as_ref().iter())
        .map(|(e1, e2)| *e1 * *e2)
        .collect()
}

pub fn vector_div<V, VC>(left: &VC, right: &VC) -> Vec<V>
where
    V: Scalar,
    VC: ValueContainer<V>,
{
    // TODO: SIMD/vectorised dispatch
    let len = left.as_ref().len();
    assert_eq!(len, right.as_ref().len());
    left.as_ref()
        .iter()
        .zip(right.as_ref().iter())
        .map(|(e1, e2)| *e1 / *e2)
        .collect()
}
