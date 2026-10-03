use lab::plus::{level_hash, shift_only_weights};
use lab::twoclass::*;

fn main() {
    let n = 10_000_000usize;
    let p: f64 = std::env::args()
        .nth(1)
        .map(|x| x.parse().unwrap())
        .unwrap_or(0.1);
    let l0: f64 = std::env::args()
        .nth(2)
        .map(|x| x.parse().unwrap())
        .unwrap_or(5.25);
    let k0 = ((1.0 - p) * n as f64) as usize;
    let mut hs: Vec<u64> = (0..k0 as u64).map(|i| level_hash(i, 0, 42)).collect();
    hs.sort_unstable();
    let c0 = ClassConf {
        lambda: l0,
        l: 512,
        r: 1,
        d: 255,
        weights: shift_only_weights(8, 512),
        window: 256,
    };
    let mut used = Bits::new(n + 64);
    let (_g, _s, _bb, bumped) = sweep(&hs, n, &c0, &mut used);
    let free: usize = (0..n).filter(|&i| !used.get(i)).count();
    println!(
        "bulk keys {k0} bumped {bumped} free {free} ({:.3}%)",
        100.0 * free as f64 / n as f64
    );
    // free density per 1/20 of the range
    let parts = 20;
    for j in 0..parts {
        let a = j * n / parts;
        let b = (j + 1) * n / parts;
        let f = (a..b).filter(|&i| !used.get(i)).count();
        print!("{:.1} ", 100.0 * f as f64 / (b - a) as f64);
    }
    println!();
    // gap length histogram (runs of occupied slots between free slots)
    let mut hist = vec![0usize; 12];
    let mut last = 0usize;
    for i in 0..n {
        if !used.get(i) {
            let g = i - last;
            last = i;
            let bin = (usize::BITS - g.leading_zeros()) as usize;
            hist[bin.min(11)] += 1;
        }
    }
    println!(
        "distance between consecutive free slots (log2 bins): {:?}",
        hist
    );
    // local density over windows of 512
    let mut dens = vec![0usize; 11];
    for w in (0..n - 512).step_by(512) {
        let f = (w..w + 512).filter(|&i| !used.get(i)).count();
        dens[(f * 10 / 512).min(10)] += 1;
    }
    println!("free-density deciles over 512-windows: {:?}", dens);
    // filler sweep
    let k1 = n - k0;
    let mut f: Vec<u64> = (0..k1 as u64)
        .map(|i| level_hash(i + 1_000_000_000, 0, 42))
        .collect();
    f.sort_unstable();
    let c1 = ClassConf {
        lambda: 1.0,
        l: 512,
        r: 1,
        d: 63,
        weights: shift_only_weights(6, 512),
        window: 256,
    };
    let (g1, s1, bb1, bumped1) = sweep(&f, n, &c1, &mut used);
    let mut by = vec![(0usize, 0usize); 10];
    for b in 0..s1.len() {
        let sz = (bb1[b + 1] - bb1[b]).min(9);
        if sz == 0 {
            continue;
        }
        by[sz].0 += 1;
        if s1[b] == 0 {
            by[sz].1 += 1;
        }
    }
    println!(
        "fillers {k1} buckets {} bumped {bumped1} by size {:?}",
        g1.buckets, by
    );
}

#[allow(dead_code)]
fn unused() {}
