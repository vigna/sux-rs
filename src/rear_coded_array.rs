use crate::traits::IndexedDict;
use num_traits::AsPrimitive;

#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Maximum block size in bytes
    pub max_block_bytes: usize,
    /// The total sum of the block size in bytes
    pub sum_block_bytes: usize,

    /// Maximum shared prefix in bytes
    pub max_lcp: usize,
    /// The total sum of the shared prefix in bytes
    pub sum_lcp: usize,

    /// maximum string length in bytes
    pub max_str_len: usize,
    /// the total sum of the strings length in bytes
    pub sum_str_len: usize,

    /// The bytes wasted writing without compression the first string in block
    pub redundancy: isize,
}

#[derive(Debug)]
pub struct RearCodedArray<Ptr: AsPrimitive<usize> = usize>
where
    usize: AsPrimitive<Ptr>,
{
    /// The encoded strings \0 terminated
    data: Vec<u8>,
    /// The pointer to in which byte the k-th string start
    pointers: Vec<Ptr>,
    /// The number of strings in a block, this regulates the compression vs
    /// decompression speed tradeoff
    k: usize,
    /// Statistics of the encoded data
    pub stats: Stats,
    /// Number of encoded strings
    len: usize,
    /// Cache of the last encoded string for incremental encoding
    last_str: Vec<u8>,
}

/// Copy a string until the first \0 from `data` to `result` and return the
/// remaining data
#[inline(always)]
fn strcpy<'a>(mut data: &'a [u8], result: &mut Vec<u8>) -> &'a [u8] {
    loop {
        let c = data[0];
        data = &data[1..];
        if c == 0 {
            break;
        }
        result.push(c);
    }
    data
}

#[inline(always)]
/// strcmp but string is a rust string and data is a \0 terminated string
fn strcmp(string: &[u8], data: &[u8]) -> core::cmp::Ordering {
    for (i, c) in string.iter().enumerate() {
        match data[i].cmp(c) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    // string has an implicit final \0
    data[string.len()].cmp(&0)
}

#[inline(always)]
/// strcmp but both string are rust strings
fn strcmp_rust(string: &[u8], other: &[u8]) -> core::cmp::Ordering {
    for (i, c) in string.iter().enumerate() {
        match other.get(i).unwrap_or(&0).cmp(c) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
    }
    // string has an implicit final \0
    other.len().cmp(&string.len())
}

impl<Ptr: AsPrimitive<usize>> RearCodedArray<Ptr>
where
    usize: AsPrimitive<Ptr>,
{
    const COMPUTE_REDUNDANCY: bool = true;

    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::with_capacity(1 << 20),
            last_str: Vec::with_capacity(1024),
            pointers: Vec::new(),
            len: 0,
            k,
            stats: Default::default(),
        }
    }

    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
        self.pointers.shrink_to_fit();
        self.last_str.shrink_to_fit();
    }

    #[inline]
    pub fn push<S: AsRef<str>>(&mut self, string: S) {
        let string = string.as_ref();
        // update stats
        self.stats.max_str_len = self.stats.max_str_len.max(string.len());
        self.stats.sum_str_len += string.len();

        // at every multiple of k we just encode the string as is
        let to_encode = if self.len % self.k == 0 {
            // compute the size in bytes of the previous block
            let last_ptr = self.pointers.last().copied().unwrap_or(0.as_());
            let block_bytes = self.data.len() - last_ptr.as_();
            // update stats
            self.stats.max_block_bytes = self.stats.max_block_bytes.max(block_bytes);
            self.stats.sum_block_bytes += block_bytes;
            // save a pointer to the start of the string
            self.pointers.push(self.data.len().as_());

            // compute the redundancy
            if Self::COMPUTE_REDUNDANCY {
                let lcp = longest_common_prefix(&self.last_str, string.as_bytes());
                let rear_length = self.last_str.len() - lcp;
                self.stats.redundancy += lcp as isize;
                self.stats.redundancy -= encode_int_len(rear_length) as isize;
            }

            // just encode the whole string
            string.as_bytes()
        } else {
            // just write the difference between the last string and the current one
            // encode only the delta
            let lcp = longest_common_prefix(&self.last_str, string.as_bytes());
            // update the stats
            self.stats.max_lcp = self.stats.max_lcp.max(lcp);
            self.stats.sum_lcp += lcp;
            // encode the len of the bytes in data
            let rear_length = self.last_str.len() - lcp;
            encode_int(rear_length, &mut self.data);
            // return the delta suffix
            &string.as_bytes()[lcp..]
        };
        // Write the data to the buffer
        self.data.extend_from_slice(to_encode);
        // push the \0 terminator
        self.data.push(0);

        // put the string as last_str for the next iteration
        self.last_str.clear();
        self.last_str.extend_from_slice(string.as_bytes());
        self.len += 1;
    }

    #[inline]
    pub fn extend<S: AsRef<str>, I: Iterator<Item = S>>(&mut self, iter: I) {
        for string in iter {
            self.push(string);
        }
    }

    /// Write the index-th string to `result` as bytes
    #[inline(always)]
    pub fn get_inplace(&self, index: usize, result: &mut Vec<u8>) {
        result.clear();
        let block = index / self.k;
        let offset = index % self.k;

        let start = self.pointers[block];
        let data = &self.data[start.as_()..];

        // decode the first string in the block
        let mut data = strcpy(data, result);

        for _ in 0..offset {
            // get how much data to throw away
            let (len, tmp) = decode_int(data);
            // throw away the data
            result.resize(result.len() - len, 0);
            // copy the new suffix
            let tmp = strcpy(tmp, result);
            data = tmp;
        }
    }

    /// Return whether the string is contained in the array.
    /// This can be used only if the strings inserted were sorted.
    pub fn contains(&self, string: &str) -> bool {
        let string = string.as_bytes();
        // first to a binary search on the blocks to find the block
        let block_idx = self
            .pointers
            .binary_search_by(|block_ptr| strcmp(string, &self.data[block_ptr.as_()..]));

        if block_idx.is_ok() {
            return true;
        }

        let mut block_idx = block_idx.unwrap_err();
        if block_idx == 0 || block_idx > self.pointers.len() {
            // the string is before the first block
            return false;
        }
        block_idx -= 1;
        // finish by a linear search on the block
        let mut result = Vec::with_capacity(self.stats.max_str_len);
        let start = self.pointers[block_idx];
        let data = &self.data[start.as_()..];

        // decode the first string in the block
        let mut data = strcpy(data, &mut result);
        let in_block = (self.k - 1).min(self.len - block_idx * self.k - 1);
        for _ in 0..in_block {
            // get how much data to throw away
            let (len, tmp) = decode_int(data);
            let lcp = result.len() - len;
            // throw away the data
            result.resize(lcp, 0);
            // copy the new suffix
            let tmp = strcpy(tmp, &mut result);
            data = tmp;

            // TODO!: this can be optimized to avoid the copy
            match strcmp_rust(string, &result) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Equal => return true,
                core::cmp::Ordering::Greater => return false,
            }
        }
        false
    }

    /// Return a sequential iterator over the strings
    pub fn iter(&self) -> RCAIter<'_, Ptr> {
        RCAIter {
            rca: self,
            index: 0,
            data: &self.data,
            buffer: Vec::with_capacity(self.stats.max_str_len),
        }
    }

    // create a sequential iterator from a given index
    pub fn iter_from(&self, index: usize) -> RCAIter<'_, Ptr> {
        let block = index / self.k;
        let offset = index % self.k;

        let start = self.pointers[block];
        let mut res = RCAIter {
            rca: self,
            index,
            data: &self.data[start.as_()..],
            buffer: Vec::with_capacity(self.stats.max_str_len),
        };
        for _ in 0..offset {
            res.next();
        }
        res
    }

    pub fn print_stats(&self) {
        println!("max_block_bytes: {}", self.stats.max_block_bytes);
        println!(
            "avg_block_bytes: {:.3}",
            self.stats.sum_block_bytes as f64 / self.len() as f64
        );

        println!("max_lcp: {}", self.stats.max_lcp);
        println!(
            "avg_lcp: {:.3}",
            self.stats.sum_lcp as f64 / self.len() as f64
        );

        println!("max_str_len: {}", self.stats.max_str_len);
        println!(
            "avg_str_len: {:.3}",
            self.stats.sum_str_len as f64 / self.len() as f64
        );

        let ptr_size: usize = self.pointers.len() * core::mem::size_of::<Ptr>();
        println!("data_bytes:  {:>15}", self.data.len());
        println!("ptrs_bytes:  {:>15}", ptr_size);

        if Self::COMPUTE_REDUNDANCY {
            println!("redundancy: {:>15}", self.stats.redundancy);

            let overhead = self.stats.redundancy + ptr_size as isize;
            println!(
                "overhead_ratio: {}",
                overhead as f64 / (overhead + self.data.len() as isize) as f64
            );
            println!(
                "no_overhead_compression_ratio: {:.3}",
                (self.data.len() as isize - self.stats.redundancy) as f64
                    / self.stats.sum_str_len as f64
            );
        }

        println!(
            "compression_ratio: {:.3}",
            (ptr_size + self.data.len()) as f64 / self.stats.sum_str_len as f64
        );
    }
}

impl<Ptr: AsPrimitive<usize>> IndexedDict for RearCodedArray<Ptr>
where
    usize: AsPrimitive<Ptr>,
{
    type Value = String;

    unsafe fn get_unchecked(&self, index: usize) -> Self::Value {
        let mut result = Vec::with_capacity(self.stats.max_str_len);
        self.get_inplace(index, &mut result);
        String::from_utf8(result).unwrap()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

pub struct RCAIter<'a, Ptr: AsPrimitive<usize>>
where
    usize: AsPrimitive<Ptr>,
{
    rca: &'a RearCodedArray<Ptr>,
    buffer: Vec<u8>,
    data: &'a [u8],
    index: usize,
}

impl<'a, Ptr: AsPrimitive<usize>> RCAIter<'a, Ptr>
where
    usize: AsPrimitive<Ptr>,
{
    pub fn new(rca: &'a RearCodedArray<Ptr>) -> Self {
        Self {
            rca,
            buffer: Vec::with_capacity(rca.stats.max_str_len),
            data: &rca.data,
            index: 0,
        }
    }
}

impl<'a, Ptr: AsPrimitive<usize>> Iterator for RCAIter<'a, Ptr>
where
    usize: AsPrimitive<Ptr>,
{
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.rca.len() {
            return None;
        }

        if self.index % self.rca.k == 0 {
            // just copy the data
            self.buffer.clear();
            self.data = strcpy(self.data, &mut self.buffer);
        } else {
            let (len, tmp) = decode_int(self.data);
            self.buffer.resize(self.buffer.len() - len, 0);
            self.data = strcpy(tmp, &mut self.buffer);
        }
        self.index += 1;

        Some(String::from_utf8(self.buffer.clone()).unwrap())
    }
}

impl<'a, Ptr: AsPrimitive<usize>> ExactSizeIterator for RCAIter<'a, Ptr>
where
    usize: AsPrimitive<Ptr>,
{
    fn len(&self) -> usize {
        self.rca.len() - self.index
    }
}

#[inline(always)]
fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
    // ofc the lcp is at most the len of the minimum string
    let min_len = a.len().min(b.len());
    // normal lcp computation
    let mut i = 0;
    while i < min_len && a[i] == b[i] {
        i += 1;
    }
    // TODO!: try to optimize with vpcmpeqb pextrb and leading count ones
    i
}

#[cfg(test)]
#[cfg_attr(test, test)]
fn test_longest_common_prefix() {
    let str1 = b"absolutely";
    let str2 = b"absorption";
    assert_eq!(longest_common_prefix(str1, str2), 4);
    assert_eq!(longest_common_prefix(str1, str1), str1.len());
    assert_eq!(longest_common_prefix(str2, str2), str2.len());
}

/// Compute the length in bytes of value encoded as VByte
#[inline(always)]
fn encode_int_len(mut value: usize) -> usize {
    let mut len = 1;
    let mut max = 1 << 7;
    while value >= max {
        len += 1;
        value -= max;
        max <<= 7;
    }
    len
}

/// VByte encode an integer
#[inline(always)]
fn encode_int(mut value: usize, data: &mut Vec<u8>) {
    let mut len = 1_usize;
    let mut max = 1 << 7;
    while value >= max {
        value -= max;
        max <<= 7;
        len += 1;
    }
    let bits_in_first = 8 - len;
    // write len - 1 in unary at the start
    let mut first = 1_u8 << bits_in_first;
    // write the lowest bits of the value
    let mask = first.saturating_sub(1);
    first |= value as u8 & mask;
    data.push(first);
    // remove the written bits
    value >>= bits_in_first;
    for _ in 0..len.saturating_sub(1) {
        data.push(value as u8);
        value >>= 8;
    }
}

#[inline(always)]
fn decode_int(data: &[u8]) -> (usize, &[u8]) {
    let len = data[0].leading_zeros() as usize + 1;
    let mut base = 0;
    // get the non-unary code bits
    let mut res = (data[0] & (0xff >> len)) as usize;
    // TODO: optimize base computation with
    // pdep((1 << len) - 1, 0x4081020408102040)
    let mut shift = 8 - len;
    for value in &data[1..len] {
        base <<= 7;
        base += 1 << 7;
        res |= (*value as usize) << shift;
        shift += 8;
    }
    (res + base, &data[len..])
}

#[cfg(test)]
#[cfg_attr(test, test)]
fn test_encode_decode_int() {
    const MAX: usize = 1 << 20;
    const MIN: usize = 0;
    let mut buffer = Vec::with_capacity(128);

    for i in MIN..MAX {
        encode_int(i, &mut buffer);
    }

    let mut data = &buffer[..];
    for i in MIN..MAX {
        let (j, tmp) = decode_int(data);
        assert_eq!(data.len() - tmp.len(), encode_int_len(i));
        data = tmp;
        assert_eq!(i, j);
    }
}
