# coalign

`coalign` is a Rust crate for fast arithmetic over keyed numeric maps.

To put it simply, this crate provides "HashMaps you can add together". Keys and values
are stored in vectors, and when two mappings contain the same keys in different
orders, `coalign` realigns values by key before applying the operation.

## What it does

- Supports elementwise `add`, `sub`, `mul`, and `div` between mappings.
- Supports scalar ops like `add_scalar`, `sub_scalar`, `mul_scalar`, and
  `div_scalar`.
- Provides in-place and allocating variants.
- Uses fast paths when vectors already share the same ordering.
- Uses SIMD-backed implementations for common primitive numeric types, with a
  generic fallback for custom scalar types.

## Why use it

`coalign` is built for cases where the key matters instead of the position of the data, and
where values may arrive in a different order.

That makes it useful for workloads like:

- keyed numeric maps with different iteration orders
- feature vectors keyed by ids
- named measurements you want to combine by key
- data pipelines that reorder rows between steps


## Example

```rust
use coalign::VectorMapping;

let a = VectorMapping::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
let b = VectorMapping::new(vec![2, 0, 1], vec![10.0, 20.0, 30.0]);

let sum = a.add(&b)?;

assert_eq!(sum.keys(), &vec![0, 1, 2]);
assert_eq!(sum.values(), &vec![21.0, 32.0, 13.0]);
# Ok::<(), coalign::errors::CoalignError>(())
```

## Scalar operations

```rust
use coalign::VectorMapping;

let values = VectorMapping::new(vec!["a", "b", "c"], vec![1.0, 2.0, 3.0]);

let out = values.mul_scalar(2.0).add_scalar(1.0);

assert_eq!(out.values(), &vec![3.0, 5.0, 7.0]);
```

Scalar operators also work on either side of a mapping:

```rust
use coalign::VectorMapping;

let values = VectorMapping::new(vec!["a", "b", "c"], vec![1.0, 2.0, 4.0]);

let right = &values + 3.0;
let left = 24.0 / &values;

assert_eq!(right.values(), &vec![4.0, 5.0, 7.0]);
assert_eq!(left.values(), &vec![24.0, 12.0, 6.0]);
```

Prefer `+`, `-`, `*`, and `/` for operator syntax. Avoid importing
`std::ops::{Add, Sub, Mul, Div}` just to write trait-method calls like
`3.0.add(mapping)`, because those trait methods can conflict with
`VectorMapping::add`, `VectorMapping::sub`, `VectorMapping::mul`, and
`VectorMapping::div`. If you need the trait-method form, prefer fully qualified
calls like `std::ops::Add::add(3.0, mapping)`.

## Compatibility model

- Same keys, same order: operate directly.
- Same keys, different order: realign by key, then operate.
- One set of keys is a subset of another: supported for mapping-to-mapping operations.
- Otherwise: return `CoalignError::IncompatibleKeys`.

## Extensibility

- Primitive integer key types are supported out of the box.
- String-like keys such as `&str`, `&[u8]`, and `Arc<str>` are supported.
- Custom key types can implement `KeyDomain`.
- Custom value types can implement `Scalar` and `ValueDomain`.

## Status

`coalign` is currently a low-level crate focused on correctness and performance
for elementwise arithmetic over labeled vectors.
