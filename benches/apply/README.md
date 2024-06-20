This bench confronts the apply function with calling get and set on each value.
To generate the data:
```$
RUSTFLAGS="-C target-cpu=native" cargo bench --bench apply 2>&1 > bench_apply.log
```
Then to plot the values:
```$
python3 plot.py
```