#![deny(unconditional_recursion)]

pub mod ranksel;
pub mod traits;

pub mod prelude {
    pub use crate::bitmap::*;
    pub use crate::compact_array::*;
    pub use crate::ranksel::prelude::*;
    pub use crate::rear_coded_array::*;
    pub use crate::traits::*;
}

mod bitmap;
mod compact_array;
mod rear_coded_array;
pub mod utils;
