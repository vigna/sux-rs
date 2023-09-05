use common_traits::*;

pub trait Bucketer {
    fn new(num_buckets: u64) -> Self;
    fn bucket(&self, hash: u64) -> usize;
    fn num_buckets(&self) -> usize;
}

pub struct UniformBucketer {
    n: usize,
}

impl Bucketer for UniformBucketer {
    #[inline(always)]
    fn new(num_buckets: u64) -> Self {
        Self {
            n: num_buckets as usize,
        }
    }
    #[inline(always)]
    fn bucket(&self, hash: u64) -> usize {
        hash.fast_range(self.n as _) as usize
    }
    #[inline(always)]
    fn num_buckets(&self) -> usize {
        self.n
    }
}

/// a = 0.6 b = 0.3
pub struct SkewedBucketer {
    num_dense_buckets: u64,
    num_sparse_buckets: u64,
}

impl Bucketer for SkewedBucketer {
    #[inline(always)]
    fn new(num_buckets: u64) -> Self {
        let num_dense_buckets = num_buckets / 3;
        Self {
            num_dense_buckets,
            num_sparse_buckets: num_buckets - num_dense_buckets,
        }
    }
    #[inline(always)]
    fn bucket(&self, hash: u64) -> usize {
        const T: u64 = (u64::MAX / 3) * 2;
        if hash < T {
            hash.fast_range(self.num_dense_buckets) as usize
        } else {
            (self.num_dense_buckets + hash.fast_range(self.num_sparse_buckets)) as usize
        }
    }
    #[inline(always)]
    fn num_buckets(&self) -> usize {
        (self.num_dense_buckets + self.num_sparse_buckets) as usize
    }
}
