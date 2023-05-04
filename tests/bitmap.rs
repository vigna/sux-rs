use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use sux::prelude::*;

#[test]
fn test_bitmap() {
    let n = 50;
    let n2 = 100;
    let u = 1000;

    let mut rng = SmallRng::seed_from_u64(0);

    let mut bm = BitMap::new(u);

    for _ in 0..10 {
        let mut values = (0..u).collect::<Vec<_>>();
        let (indices, _) = values.partial_shuffle(&mut rng, n2);

        for i in indices[..n].iter().copied() {
            bm.set(i as usize, 1).unwrap();
        }
        
        for i in 0..u {
            assert_eq!(bm.get(i).unwrap() != 0, indices[..n].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i as usize, 1).unwrap();
        }
        
        for i in 0..u {
            assert_eq!(bm.get(i).unwrap() != 0, indices.contains(&i));
        }

        for i in indices[..n].iter().copied() {
            bm.set(i as usize, 0).unwrap();
        }

        for i in 0..u {
            assert_eq!(bm.get(i).unwrap() != 0, indices[n..].contains(&i));
        }

        for i in indices[n..].iter().copied() {
            bm.set(i as usize, 0).unwrap();
        }

        for i in 0..u {
            assert_eq!(bm.get(i).unwrap(), 0);
        }
    }
}