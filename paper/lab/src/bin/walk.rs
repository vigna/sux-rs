//! Runs PHast-R placement with unbounded repair (see `lab::walk`).
//!
//! Usage: walk <keys> <cap> <instances> <spec>...
//!
//! A spec is `<S>:<log2 L>:<log2 R>:<lambda>:<alpha>[:prio][:a<age cost>]`;
//! with `prio`, buckets are processed by PHast-R priority instead of in index
//! order; `a<age cost>` sets the cost per slot of the age of evicted buckets;
//! `o<owner cost>` sets the cost per evicted bucket; `z<zone>` forbids
//! evicting buckets more than `<zone>` slots behind the frontier; `single`
//! evicts at most one bucket per placement.
//! Statistics are summed over independent instances.
use lab::dfs::Keys;
use lab::walk::{WalkConf, WalkStats, run};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args[1].parse::<usize>().unwrap() / 64 * 64;
    let cap: u64 = args[2].parse().unwrap();
    let inst: u64 = args[3].parse().unwrap();
    for spec in &args[4..] {
        let p: Vec<&str> = spec.split(':').collect();
        let c = WalkConf {
            seed_bits: p[0].parse().unwrap(),
            log2_slice_len: p[1].parse().unwrap(),
            log2_layouts: p[2].parse().unwrap(),
            priority: p[5..].contains(&"prio"),
            cap,
            age_cost: p[5..]
                .iter()
                .find_map(|x| x.strip_prefix('a').and_then(|y| y.parse().ok()))
                .unwrap_or(0),
            zone: p[5..]
                .iter()
                .find_map(|x| x.strip_prefix('z').and_then(|y| y.parse().ok()))
                .unwrap_or(0),
            single_owner: p[5..].contains(&"single"),
            owner_cost: p[5..]
                .iter()
                .find_map(|x| x.strip_prefix('o').and_then(|y| y.parse().ok()))
                .unwrap_or(0),
        };
        let lambda: f64 = p[3].parse().unwrap();
        let alpha: f64 = p[4].parse().unwrap();
        let w = (1usize << c.log2_slice_len) + ((1usize << c.seed_bits) >> c.log2_layouts) - 1;
        let mut tot = WalkStats::default();
        let (mut keys_tot, mut range_max, mut range_sum, mut failed) = (0usize, 0usize, 0usize, 0);
        let mut bits = 0.0;
        for i in 0..inst {
            let keys = Keys::with_load(n, lambda, alpha, i);
            let r = keys.bridge_range();
            range_max = range_max.max(r);
            range_sum += r;
            let st = run(&keys, &c);
            keys_tot += n;
            bits += c.seed_bits as f64 * keys.buckets as f64;
            failed += (st.bumped_keys > 0) as usize;
            tot.layers += st.layers;
            tot.episodes += st.episodes;
            tot.evictions += st.evictions;
            tot.direct_replacements += st.direct_replacements;
            tot.max_episode = tot.max_episode.max(st.max_episode);
            for (a, b) in tot.episode_hist.iter_mut().zip(st.episode_hist.iter()) {
                *a += b;
            }
            tot.bumped_keys += st.bumped_keys;
            for (a, b) in tot.capped_by_pos.iter_mut().zip(st.capped_by_pos.iter()) {
                *a += b;
            }
            tot.holes += st.holes;
            tot.ns_per_key += st.ns_per_key / inst as f64;
        }
        let hist: Vec<String> = tot
            .episode_hist
            .iter()
            .enumerate()
            .filter(|x| *x.1 != 0)
            .map(|(i, x)| format!("{}:{}", 1u64 << i, x))
            .collect();
        println!(
            "{spec:20} W {w:4} range avg {:6.0} max {range_max:6}  {:.4} b/k  epis/b {:.4}  ev/b {:8.4}  direct {:.3}  maxep {:8}  bumped {:.5}%  holes {:.5}%  failed {failed}/{inst}  {:6.0} ns/k  capped@{:?}  [{}]",
            range_sum as f64 / inst as f64,
            bits / keys_tot as f64,
            tot.episodes as f64 / tot.layers as f64,
            tot.evictions as f64 / tot.layers as f64,
            tot.direct_replacements as f64 / tot.evictions.max(1) as f64,
            tot.max_episode,
            100.0 * tot.bumped_keys as f64 / keys_tot as f64,
            100.0 * tot.holes as f64 / keys_tot as f64,
            tot.ns_per_key,
            tot.capped_by_pos,
            hist.join(" ")
        );
    }
}
