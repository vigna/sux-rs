use std::iter::zip;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::{bits::BitVec, rank_sel::SimpleSelect};

fn main() {
    let lens = [
        1_000_000,
        3_000_000,
        10_000_000,
        30_000_000,
        100_000_000,
        300_000_000,
        1_000_000_000,
    ];
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

    let wd = std::env::current_dir().unwrap().display().to_string();
    let source_path = format!("{}/target/criterion/select_fixed2/", wd);
    let target_path = format!("{}/target/criterion/fixed_filtered/", wd);

    for (b, (len, d)) in zip(&bitvecs, &bitvec_ids) {
        let simple = SimpleSelect::new(b.clone(), 3);
        let log2_ones_per_inventory = simple.log2_ones_per_inventory();
        let log2_u64_per_subinventory = simple.log2_u64_per_subinventory();
        let source_folder_name = format!(
            "{}_{}_{}_{}_0/",
            log2_ones_per_inventory, log2_u64_per_subinventory, len, d
        );
        let target_folder_name = format!("{}_{}_0", len, d);
        let source = format!("{}{}", source_path, source_folder_name);
        let target = format!("{}{}", target_path, target_folder_name);
        std::fs::rename(source, target).unwrap();
    }
}
