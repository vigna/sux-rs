use std::iter::zip;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::{bits::BitVec, rank_sel::SimpleSelect};

fn main() {
    let lens = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000];
    let densities = [0.2, 0.5, 0.8];

    let mut bitvecs = Vec::<BitVec>::new();
    let mut bitvec_ids = Vec::<(u64, f64)>::new();
    let mut rng = SmallRng::seed_from_u64(0);
    for len in lens {
        for density in densities {
            let bitvec = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
            bitvecs.push(bitvec);
            bitvec_ids.push((len, density));
        }
    }

    for (b, (len, d)) in zip(&bitvecs, &bitvec_ids) {
        let simple = SimpleSelect::new(b.clone(), 3);
        println!("--- bitvec with len: {}, density: {}", len, d);
        let log2_ones_per_inventory = simple.get_log2_ones_per_inventory();
        let log2_u64_per_subinventory = simple.get_log2_u64_per_subinventory();
        println!("log2_ones_per_inventory: {}", log2_ones_per_inventory);
        println!("log2_u64_per_subinventory: {}", log2_u64_per_subinventory);
    }
}
