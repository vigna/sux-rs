/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Main traits used in the implementation of succinct and compressed data
//! structures.

pub mod bal_paren;
pub use bal_paren::*;

pub mod bit_field_slice;
use std::{rc::Rc, sync::Arc};

#[allow(unused_imports)]
use crate::bits::bit_vec::BitVec;
use ambassador::delegatable_trait;
#[allow(unused_imports)]
use atomic_primitive::PrimitiveAtomicUnsigned;
pub use bit_field_slice::*;

pub mod indexed_dict;
use impl_tools::autoimpl;
pub use indexed_dict::*;

pub mod iter;
pub use iter::*;

pub mod rank_sel;
use num_primitive::PrimitiveUnsigned;
pub use rank_sel::*;

pub mod bit_vec_ops;
pub use bit_vec_ops::*;

/// Error returned by [`TryIntoUnaligned::try_into_unaligned`] when the
/// bit width does not satisfy the constraints for unaligned reads.
#[derive(Debug)]
pub struct UnalignedConversionError(pub String);

impl std::fmt::Display for UnalignedConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnalignedConversionError {}

/// A convenient alias for the unaligned variant of a type that implements
/// [`TryIntoUnaligned`].
pub type Unaligned<T> = <T as TryIntoUnaligned>::Unaligned;

/// A trait for types that can be converted into an unaligned variant
/// that uses branchless [unaligned reads].
///
/// The conversion will fail if the bit width for which unaligned reads are
/// required does not satisfy the constraints for the word type used by the
/// structure or its components.
///
/// Compound types (e.g., [`SignedFunc`]) implement this trait by recursively
/// converting their inner components.
///
/// You can obtain an unaligned variant of a structure just by chaining the
/// [`try_into_unaligned`] method at construction time. Since the unaligned
/// variant of a type can be quite complicated, there is an [`Unaligned`]
/// type alias to refer to it more conveniently. For example,
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::LcpMmphfInt;
/// # use sux::utils::FromSlice;
/// # use sux::traits::{Unaligned, TryIntoUnaligned};
/// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
/// let unaligned_func: Unaligned<LcpMmphfInt<u64>> =
///     LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// Or, equivalently, without chaining:
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::LcpMmphfInt;
/// # use sux::utils::FromSlice;
/// # use sux::traits::TryIntoUnaligned;
/// # let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
/// let func: LcpMmphfInt<u64> =
///     LcpMmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
/// let unaligned_func = func.try_into_unaligned()?;
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// [unaligned reads]: crate::bits::BitFieldVec::get_unaligned
/// [`SignedFunc`]: crate::func::SignedFunc
/// [`try_into_unaligned`]: Self::try_into_unaligned
pub trait TryIntoUnaligned {
    /// The unaligned version of this type.
    type Unaligned;
    /// Converts `self` into the unaligned variant.
    ///
    /// # Errors
    ///
    /// Returns an error if the bit width for which unaligned reads are required
    /// does not satisfy the constraints for the word type used by the structure
    /// or its components.
    fn try_into_unaligned(self) -> Result<Self::Unaligned, UnalignedConversionError>;
}

/// A trait for primitive types that can be used in [backends].
///
/// This trait is equivalent to [`PrimitiveUnsigned`], but it has a shorter
/// name and provides constants [`ZERO`] and [`ONE`], which avoid a dependency
/// on the [`num-traits`] crate.
///
/// [`num-traits`]: https://crates.io/crates/num-traits
/// [backends]: crate::traits::Backend
/// [`ZERO`]: Self::ZERO
/// [`ONE`]: Self::ONE
pub trait Word: PrimitiveUnsigned {
    const ZERO: Self;
    const ONE: Self;
}

// Note: once we can switch to a more recent num-primitive,
// we will be able to use CONST[0] and CONST[1] to define ZERO and ONE
// and we will be able to remove this macro.
macro_rules! impl_word {
    ($($ty:ty),*) => {
        $(impl Word for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
        })*
    };
}

impl_word!(u8, u16, u32, u64, u128, usize);

/// The basic trait defining backends.
///
/// Backends are types used as underlying storage by other types. Backends are
/// made of contiguous sequences of words, and the type of such words is given
/// by the [`Backend::Word`] associated type.
///
/// For example, the type `BitVec<Vec<usize>>` represents a bit vector using a
/// vector of `usize` as its backend.
///
/// This trait provides no methods: access to the underlying data is provided by
/// other traits, such as [`AsRef`] or [`AsMut`]. Moreover, the [`Backend::Word`]
/// associated type is usually bound with an appropriate trait such as [`Word`] or
/// [`PrimitiveAtomicUnsigned`].
///
/// For example, the typical read-only word-based backend `B` for bit vectors
/// and vectors of bit fields satisfies the bound
///
/// ```ignore
/// B: Backend<Word: Word> + AsRef<[B::Word]>
/// ```
///
/// whereas an analogous atomic backend satisfies
///
/// ```ignore
/// B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>
/// ```
///
/// Bit-based backends satisfy also the [`BitLength`] trait, which specifies the
/// number of valid bits in the backend. For example, [`BitVec`]'s only
/// parameter is a word-based backend, but [`BitVec`] itself is a bit-based
/// backend, and thus it implements [`BitLength`] and delegates [`Backend`],
/// [`AsRef`] and [`AsMut`] to its word-based backend.
///
/// Note that *traits* manipulating backends such as [`BitVecOps`] do not use
/// this trait, but are rather parametrized by a word type `W`, and extend
/// traits such as [`AsRef<[W]>`] as needed.
///
/// However, *types* with a backend need this trait to avoid a redundant
/// specification of the word type in isolation and as part of the backend. If
/// we did not have this trait to specify the word type, we would need to write
/// something like `BitVec<u64, Vec<u64>>`, because there is no way to infer `W`
/// from `Vec<W>` when `Vec<W>` is an opaque type parameter, and in an `impl`
/// block a syntax like `impl<W, B: AsRef<[W]>>` will not compile unless the
/// type (or the implemented trait) contains `W`.
///
/// This trait is delegated by every
/// [rank/select structure](crate::rank_sel) to its backend (an inner
/// field) together with [`AsRef`] and [`BitLength`] so that, for
/// example, a rank/select structure can be used as a backend for another
/// structure without any boilerplate.
///
/// We implement this trait for slices, vectors, and arrays; moreover, we
/// delegate it automatically to references, boxed types, and reference-counted
/// wrappers.
///
/// [`AsRef`]: core::convert::AsRef
/// [`AsRef<[W]>`]: core::convert::AsRef
/// [`AsMut`]: core::convert::AsMut
/// [`Backend::Word`]: Self::Word
/// [`BitVec<Vec<usize>>`]: BitVec
/// [`Word`]: crate::traits::Word
/// [`BitLength`]: crate::traits::BitLength
/// [`BitVec`]: BitVec
/// [`BitVecOps`]: crate::traits::BitVecOps
/// [`PrimitiveAtomicUnsigned`]: PrimitiveAtomicUnsigned
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>, Rc<T>, Arc<T>)]
#[delegatable_trait]
pub trait Backend {
    /// The word type used by this backend.
    type Word;
}

impl<W> Backend for [W] {
    type Word = W;
}

impl<W> Backend for Vec<W> {
    type Word = W;
}

impl<W, const N: usize> Backend for [W; N] {
    type Word = W;
}
