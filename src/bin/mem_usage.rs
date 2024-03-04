use mem_dbg::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::{
    bits::BitVec,
    rank_sel::{Rank9Sel, SimpleSelect},
    traits::Select,
};

trait SelStruct<B>: Select {
    fn new(bits: B) -> Self;
}
impl SelStruct<BitVec> for SimpleSelect {
    fn new(bits: BitVec) -> Self {
        SimpleSelect::new(bits, 3)
    }
}
impl SelStruct<BitVec> for Rank9Sel {
    fn new(bits: BitVec) -> Self {
        Rank9Sel::new(bits)
    }
}

fn show_mem<S: SelStruct<BitVec> + MemSize + MemDbg>(uniform: bool) {
    let mut rng = SmallRng::seed_from_u64(0);
    let density = 0.5;
    let len = 10_000_000;
    let (density0, density1) = if uniform {
        (density, density)
    } else {
        (density * 0.01, density * 0.99)
    };

    let first_half = loop {
        let b = (0..len / 2)
            .map(|_| rng.gen_bool(density0))
            .collect::<BitVec>();
        if b.count_ones() > 0 {
            break b;
        }
    };
    let second_half = (0..len / 2)
        .map(|_| rng.gen_bool(density1))
        .collect::<BitVec>();

    let bits = first_half
        .into_iter()
        .chain(second_half.into_iter())
        .collect::<BitVec>();

    let rank9sel: S = S::new(bits);

    println!("size:     {}", rank9sel.mem_size(SizeFlags::default()));
    println!("capacity: {}", rank9sel.mem_size(SizeFlags::CAPACITY));

    rank9sel.mem_dbg(DbgFlags::default()).unwrap();
}

fn main() {
    show_mem::<SimpleSelect>(true);
    show_mem::<Rank9Sel>(true);
    show_mem::<SimpleSelect>(false);
    show_mem::<Rank9Sel>(false);
}
