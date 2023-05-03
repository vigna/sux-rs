#![deny(unconditional_recursion)]

pub mod traits;
pub mod ranksel;

pub mod prelude {
    pub use crate::traits::*;
    pub use crate::ranksel::prelude::*;
    pub use crate::bitmap::*;
    pub use crate::compact_array::*;
}

mod bitmap;
mod compact_array;
pub(crate) mod utils;
