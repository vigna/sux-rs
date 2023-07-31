mod to_lebe;
pub use to_lebe::*;

/// Compute the padding needed for alignement, i.e., the number so that
/// `((value + pad_align_to(value, bits) & (bits - 1) == 0`.
///
/// ```
/// use sux::utils::pad_align_to;
/// assert_eq!(7 + pad_align_to(7, 8), 8);
/// assert_eq!(8 + pad_align_to(8, 8), 8);
/// assert_eq!(9 + pad_align_to(9, 8), 16);
/// ```
pub fn pad_align_to(value: usize, bits: usize) -> usize {
    value.wrapping_neg() & (bits - 1)
}
