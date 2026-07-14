# Optimization Log

## 2026-07-13: Jacobson pioneer construction

### Target

`src/bal_paren/jacobson.rs::build_pioneers`

Fully nested balanced-parentheses inputs exposed quadratic construction: each far
opening scanned the block-count array from its own block until it found the next
block with unmatched far closes.

### Environment

- CPU: AMD Ryzen 9 7950X3D
- OS: Linux 6.18.37 x86_64
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`
- Profile: Cargo `bench` (`--release`, optimized with debug information)
- Criterion: 10 samples, 0.2 s warm-up, 0.5 s measurement

### Command

```console
cargo bench --bench bench_jacobson -- --sample-size 10 --warm-up-time 0.2 --measurement-time 0.5
```

### Results

Times are Criterion 95% confidence intervals. Input size is the number of pairs
in a fully nested sequence.

| Pairs | Before | After | Criterion change | Midpoint speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 79.022-79.859 us | 18.411-18.506 us | -76.724% | 4.30x |
| 8,192 | 284.26-286.74 us | 36.541-36.689 us | -87.134% | 7.80x |
| 16,384 | 1.0790-1.0952 ms | 73.501-73.786 us | -93.259% | 14.75x |

All three changes were statistically significant (`p = 0.00 < 0.05`). The new
throughput remains approximately constant at 443-448 million parentheses per
second across the measured sizes; the old throughput fell from 103 to 30
million parentheses per second as the input doubled.

### Change

Replace the per-opening linear search with a stack of processed blocks that
still contain unmatched far closes. Because blocks are scanned right-to-left,
the nearest matching block is always the stack's last element. Each block is
pushed and popped once, reducing pioneer construction from quadratic to linear
time on the adversarial nested shape without changing the pioneer-selection
rule.

### Verification

```console
cargo test --lib bal_paren::jacobson
cargo test --test test_jacobson
```

Results: 14 library tests and 16 integration tests passed.
