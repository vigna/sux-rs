#![deny(unconditional_recursion)]

pub mod traits;

pub mod prelude {
    pub use crate::traits::*;
}

mod bitmap;
pub(crate) mod utils;
