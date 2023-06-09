pub mod elias_fano;
pub mod sparse_index;
pub mod sparse_zero_index;

pub mod prelude {
    pub use super::elias_fano::*;
    pub use super::sparse_index::*;
    pub use super::sparse_zero_index::*;
}
