#[derive(Debug, thiserror::Error)]
pub enum CoalignError {
    #[error("Cannot add two VectorMappings with incompatible keys")]
    IncompatibleKeys,
    #[error("KeyError")]
    KeyError,
}
