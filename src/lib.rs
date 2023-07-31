#![deny(unconditional_recursion)]

pub mod hash;
pub mod mph;
pub mod ranksel;
pub mod traits;

pub mod prelude {
    pub use crate::bitmap::*;
    pub use crate::compact_array::*;
    pub use crate::hash::*;
    pub use crate::mph::*;
    pub use crate::ranksel::prelude::*;
    pub use crate::rear_coded_list::*;
    pub use crate::traits::*;
    pub use crate::word_array::*;
}

pub mod bitmap;
pub mod compact_array;
pub mod rear_coded_list;
pub mod utils;
pub mod word_array;
