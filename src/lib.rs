pub mod traits;

pub mod prelude {
    pub use crate::traits::*;
}

mod bitmap;
pub use bitmap::BitMap;
