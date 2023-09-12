use rand::rngs::SmallRng;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    const N: usize = 1024 * 1024 * 1024;
    let mut data = Vec::<u64>::new();
    data.resize(N, 0);
    let fill = data.as_mut_slice();

    let mut r = StdRng::seed_from_u64(0);
    let start = Instant::now();
    r.fill(fill);
    black_box(());
    println!("ChaCha12 {}", start.elapsed().as_nanos() as f64 / N as f64);

    let mut r = ChaCha8Rng::seed_from_u64(0);
    let start = Instant::now();
    r.fill(fill);
    black_box(());
    println!("ChaCha8 {}", start.elapsed().as_nanos() as f64 / N as f64);

    let mut r = SmallRng::seed_from_u64(0);
    let start = Instant::now();
    r.fill(fill);
    black_box(());

    println!(
        "xoshiro256++ {}",
        start.elapsed().as_nanos() as f64 / N as f64
    );
}
