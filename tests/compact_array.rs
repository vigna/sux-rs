use core::sync::atomic::{AtomicU64, Ordering};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use sux::prelude::*;

#[test]
fn test_compact_array() {
    for bit_width in 0..64 {
        let n = 100;
        let u = 1 << bit_width;
        let mut rng = SmallRng::seed_from_u64(0);

        let mut cp = CompactArray::new(bit_width, n);
        for _ in 0..10 {
            let values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                cp.set(i, values[i]);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(cp.get(i), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(cp.get(i), values[i]);
            }
        }
        // convert to atomic
        let cp: CompactArray<Vec<AtomicU64>> = cp.into();
        for _ in 0..10 {
            let values = (0..n).map(|_| rng.gen_range(0..u)).collect::<Vec<_>>();

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                cp.set_atomic(i, values[i], Ordering::Relaxed);
            }

            for (i, value) in values.iter().enumerate() {
                assert_eq!(cp.get_atomic(i, Ordering::Relaxed), *value);
            }

            let mut indices = (0..n).collect::<Vec<_>>();
            indices.shuffle(&mut rng);

            for i in indices {
                assert_eq!(cp.get_atomic(i, Ordering::Relaxed), values[i]);
            }
        }
    }
}
