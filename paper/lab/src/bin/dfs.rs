//! Runs the CONSENSUS-style search for bump-free placements (see `lab::dfs`).
//!
//! Usage: dfs <keys> <budget per bucket> <spec>...
//!
//! A spec is `<S>:<log2 L>:<log2 R>:<lambda>:<chain>[:seed][:<seed>]`, where
//! `<chain>` is the number of previous seeds hashed into the offsets (0: no
//! chaining) and `seed` tries seeds in seed order instead of by increasing
//! sum of positions. The last field, if numeric, is the random seed.
use lab::dfs::{DfsConf, Keys, run, verify};
use std::f64::consts::LOG2_E;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args[1].parse::<usize>().unwrap() / 64 * 64;
    let budget: u64 = args[2].parse().unwrap();
    for spec in &args[3..] {
        let p: Vec<&str> = spec.split(':').collect();
        let c = DfsConf {
            seed_bits: p[0].parse().unwrap(),
            log2_slice_len: p[1].parse().unwrap(),
            log2_layouts: p[2].parse().unwrap(),
            lambda: p[3].parse().unwrap(),
            chain: p[4].parse().unwrap(),
            min_sum: !p[5..].contains(&"seed"),
            max_nodes_per_bucket: budget,
        };
        let rs: u64 = p[5..].iter().find_map(|x| x.parse().ok()).unwrap_or(0);
        let keys = Keys::new(n, c.lambda, rs);
        let (st, seeds) = run(&keys, &c);
        let ok = st.success && verify(&keys, &c, &seeds);
        let b = keys.buckets as f64;
        let eps = c.seed_bits as f64 - n as f64 / b * LOG2_E;
        let lb = st.layers as f64;
        let hist: Vec<String> = st
            .retreat_hist
            .iter()
            .enumerate()
            .filter(|x| *x.1 != 0)
            .map(|(i, x)| format!("{}:{}", 1 << i, x))
            .collect();
        println!(
            "{spec:22} {:.4} b/k eps {eps:+.3}  {} deepest {:8} nodes/b {:8.3}  srch/b {:8.3}  back/b {:8.4}  maxret {:6}  end {:9}  {:7.0} ns/k  ret [{}]",
            c.seed_bits as f64 * b / n as f64,
            if ok {
                "OK  "
            } else if st.success {
                "BAD "
            } else {
                "FAIL"
            },
            st.deepest,
            st.nodes as f64 / lb,
            st.searches as f64 / lb,
            st.backtracks as f64 / lb,
            st.max_retreat,
            st.endgame_nodes,
            st.ns_per_key,
            hist.join(" ")
        );
    }
}
