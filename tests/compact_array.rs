use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;
use rand::seq::SliceRandom;
use sux::prelude::*;


#[test]
fn test_compact_array() {
    let bit_width = 7;
    let n = 100;
    let u = 1 << bit_width;
    let mut rng = SmallRng::seed_from_u64(0);

    let mut cp = CompactArray::new(bit_width, n);
    for _ in 0..10 {
        let values = (0..n).map(|_| {
            rng.gen_range(0..u)
        }).collect::<Vec<_>>();

        let mut indices = (0..n).collect::<Vec<_>>();
        indices.shuffle(&mut rng);

        for i in indices {
            cp.set(i, values[i]).unwrap();
        }
        
        for (i, value) in values.iter().enumerate() {
            assert_eq!(cp.get(i).unwrap(), *value);
        }

        let mut indices = (0..n).collect::<Vec<_>>();
        indices.shuffle(&mut rng);

        for i in indices {
            assert_eq!(cp.get(i).unwrap(), values[i]);
        }
    }
}