/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Immutable lists of strings compressed by prefix omission via rear coding and
supports generic encodings.

*/

use crate::prelude::*;
use anyhow::Result;
use epserde::*;
use lender::for_;
use lender::{ExactSizeLender, IntoLender, Lender, Lending};
use mem_dbg::*;

#[allow(clippy::len_without_is_empty)]
/// Encoder needed to write a CompressedRearCodedList
pub trait Encoder {
    /// Encodes the rear coded length of the lcp of the string and returns
    /// the number of bits written.
    fn encode_length(&mut self, len: usize) -> usize;
    /// Encode the string bytes and return the number of bits written.
    fn encode_bytes(&mut self, data: &[u8]) -> usize;
    /// Push the block offset to the offsets builder.
    fn push_block_offset(&mut self);

    /// The type of the decoder factory that will be created from the encoder.
    type DecoderFactory: DecoderFactory;
    /// To build the CRCL from the builder, we need to create a decoder factory.
    /// This method consumes the encoder and returns the decoder factory.
    fn into_decoder_factory(self) -> Result<Self::DecoderFactory>;
}

/// Decoder factory needed to create a Decoder
pub trait DecoderFactory {
    /// The Decoder returned
    type Decoder<'a>: Decoder
    where
        Self: 'a;
    /// Returns a new decoder for the block at index `block_idx`.
    fn get_decoder(&self, block_idx: usize) -> Result<Self::Decoder<'_>>;
}

/// Decoder needed to read a CompressedRearCodedList
pub trait Decoder {
    /// Decode the rear coded length of the lcp of the string.
    fn decode_length(&mut self) -> usize;
    /// Decode the string bytes and append them to `data`.
    fn decode_bytes(&mut self, data: &mut Vec<u8>);
}

#[allow(clippy::len_without_is_empty)]
/// Trait for a builder that will be used for the offset of blocks in the CRCL.
pub trait OffsetsBuilder {
    /// Push the block offset to the offsets builder.
    fn push_offset(&mut self, block_offset: usize);
    /// Return the number of offsets pushed.
    fn len(&self) -> usize;

    /// The type of the offsets that will be created from the builder.
    type Offset: IndexedDict<Input = usize, Output = usize>;
    /// To build the offsets from the builder, we need to create a dictionary.
    fn into_offsets(self) -> Result<Self::Offset>;
}

/// Encode gaps in offsets with gamma codes while building, and then convert
/// to Elias-Fano.
pub mod ef {
    use crate::prelude::*;
    use anyhow::Result;
    use dsi_bitstream::prelude::*;
    use std::io::Seek;

    pub type EF = EliasFano<SelectFixed2>;
    pub type Writer<W> = BufBitWriter<BigEndian, WordAdapter<u32, W>>;
    pub type Reader<R> = BufBitReader<BigEndian, WordAdapter<u32, R>>;

    pub struct OffsetsBuilder {
        gaps: Writer<std::io::Cursor<Vec<u8>>>,
        last_offset: usize,
        len: usize,
    }

    impl core::default::Default for OffsetsBuilder {
        #[inline(always)]
        fn default() -> Self {
            Self {
                gaps: BufBitWriter::new(WordAdapter::new(std::io::Cursor::new(
                    Vec::with_capacity(128),
                ))),
                last_offset: 0,
                len: 0,
            }
        }
    }

    impl super::OffsetsBuilder for OffsetsBuilder {
        #[inline(always)]
        fn len(&self) -> usize {
            self.len
        }
        #[inline(always)]
        fn push_offset(&mut self, block_offset: usize) {
            let gap = block_offset - self.last_offset;
            self.gaps.write_gamma(gap as u64).unwrap(); // can't be zero
            self.last_offset = block_offset;
            self.len += 1;
        }
        type Offset = EF;
        #[inline(always)]
        fn into_offsets(self) -> Result<Self::Offset> {
            let mut builder = EliasFanoBuilder::new(self.len, self.last_offset);
            let mut last_offset = 0;
            let mut data = self.gaps.into_inner()?.into_inner();
            data.seek(std::io::SeekFrom::Start(0)).unwrap();
            let mut reader: Reader<std::io::Cursor<Vec<u8>>> =
                BufBitReader::new(WordAdapter::new(data));
            for _ in 0..self.len {
                let gap = reader.read_gamma()?;
                last_offset += gap;
                builder.push(last_offset as usize)?;
            }
            builder.build().convert_to()
        }
    }
}

/// Encode lengths with gamma codes, then encode each byte with optimal
/// huffman codes.
pub mod huffman {
    use crate::prelude::*;
    use anyhow::Result;
    use dsi_bitstream::prelude::huffman::HuffmanTree;
    use dsi_bitstream::prelude::*;
    use epserde::Epserde;
    use mem_dbg::{MemDbg, MemSize};

    pub type Writer<W> = BufBitWriter<BigEndian, WordAdapter<u32, W>>;
    pub type Reader<R> = BufBitReader<BigEndian, WordAdapter<u32, R>>;

    pub struct Encoder<OB: super::OffsetsBuilder = super::ef::OffsetsBuilder> {
        offsets_builder: OB,
        writer: Writer<std::io::Cursor<Vec<u8>>>,
        huffman: HuffmanTree,
        counts: [usize; 256],
        len: usize,
    }

    impl<OB: super::OffsetsBuilder> Encoder<OB> {
        #[inline(always)]
        pub fn new(huffman: HuffmanTree, offsets_builder: OB) -> Self {
            Self {
                offsets_builder,
                writer: BufBitWriter::new(WordAdapter::new(std::io::Cursor::new(
                    Vec::with_capacity(128),
                ))),
                huffman,
                counts: [0; 256],
                len: 0,
            }
        }
    }

    impl<OB: super::OffsetsBuilder> super::Encoder for Encoder<OB> {
        #[inline(always)]
        fn encode_length(&mut self, len: usize) -> usize {
            let res = self.writer.write_gamma(len as u64).unwrap();
            self.len += res;
            res
        }
        fn push_block_offset(&mut self) {
            self.offsets_builder.push_offset(self.len);
        }
        #[inline(always)]
        fn encode_bytes(&mut self, data: &[u8]) -> usize {
            let mut bits_written = 0;
            bits_written += self.writer.write_gamma(data.len() as u64).unwrap();
            for &byte in data {
                bits_written += self.huffman.encode(byte as _, &mut self.writer).unwrap();
                self.counts[byte as usize] += 1;
            }
            self.len += bits_written;
            bits_written
        }
        type DecoderFactory = DecoderFactory<OB::Offset>;
        #[inline(always)]
        fn into_decoder_factory(self) -> Result<Self::DecoderFactory> {
            Ok(DecoderFactory {
                offsets: self.offsets_builder.into_offsets()?,
                data: self.writer.into_inner()?.into_inner().into_inner(),
                huffman: self.huffman,
            })
        }
    }

    #[derive(Debug, Epserde, MemDbg, MemSize)]
    pub struct DecoderFactory<O: IndexedDict<Input = usize, Output = usize> = super::ef::EF> {
        offsets: O,
        data: Vec<u8>,
        huffman: HuffmanTree,
    }

    impl<O: IndexedDict<Input = usize, Output = usize>> super::DecoderFactory for DecoderFactory<O> {
        type Decoder<'a> = Decoder<'a> where Self: 'a, O: 'a;
        #[inline(always)]
        fn get_decoder(&self, block_idx: usize) -> Result<Self::Decoder<'_>> {
            let mut reader =
                BufBitReader::new(WordAdapter::new(std::io::Cursor::new(self.data.as_slice())));
            reader
                .set_bit_pos(self.offsets.get(block_idx) as u64)
                .unwrap();
            Ok(Decoder {
                reader,
                huffman: &self.huffman,
            })
        }
    }

    pub struct Decoder<'a> {
        reader: Reader<std::io::Cursor<&'a [u8]>>,
        huffman: &'a HuffmanTree,
    }

    impl<'a> super::Decoder for Decoder<'a> {
        #[inline(always)]
        fn decode_length(&mut self) -> usize {
            self.reader.read_gamma().unwrap() as usize
        }
        #[inline(always)]
        fn decode_bytes(&mut self, data: &mut Vec<u8>) {
            let len = self.reader.read_gamma().unwrap();
            for _ in 0..len {
                data.push(self.huffman.decode(&mut self.reader).unwrap() as u8);
            }
        }
    }
}

/// The count of occurences of each byte, computed /usr of my linux filesystem.
pub const DATA_COUNTS: [usize; 256] = [
    5948740863, 470157844, 244335113, 175340730, 242746957, 147627855, 117905741, 102849596,
    254466929, 58803884, 72725791, 61087764, 73500788, 49865552, 240726723, 537829375, 204691693,
    56549435, 43709136, 30828876, 54564741, 127435143, 29060865, 27022401, 115521953, 25512714,
    22515862, 23749293, 37804277, 22690597, 47159611, 112862399, 233418066, 31600590, 26396567,
    22081700, 406207766, 49848215, 20868877, 21033828, 109031531, 58621773, 23188149, 28259325,
    36364848, 31128034, 67245703, 34067957, 106921614, 125418362, 42939663, 34385156, 44488680,
    45020239, 29207572, 25770313, 76573115, 93723595, 33147566, 32905643, 44602452, 37491413,
    22251340, 24565514, 109586208, 265555027, 85965062, 63048024, 221132535, 139662447, 47209016,
    50443765, 980961711, 182482091, 28123174, 30301706, 284688353, 83182955, 47246162, 30515337,
    95123802, 28342457, 35589800, 73404839, 80417545, 67621151, 37836903, 34245599, 49443492,
    23357774, 29301494, 41963134, 56388302, 56283573, 28452706, 123544867, 51761257, 118288986,
    45053121, 88662519, 91505023, 181092237, 176099237, 59682936, 61100072, 110615433, 24807458,
    30183554, 100910599, 59199677, 112507609, 129474453, 97805787, 17983461, 124068483, 104002251,
    255763472, 110838547, 47732715, 31291463, 61107590, 40830411, 20113473, 24599424, 56352918,
    42495518, 27342645, 29869998, 100007073, 38470896, 21528775, 210526018, 176976628, 174758205,
    35976983, 25119328, 48957611, 554790297, 14453470, 436477030, 40171412, 231584701, 29192425,
    22829915, 70929993, 17196393, 14497654, 18221749, 30555186, 22850103, 13529504, 13744010,
    31263260, 13890540, 12819117, 13049525, 23605287, 15933384, 12562855, 18207325, 40655903,
    13681238, 12659588, 14984124, 19128117, 14342876, 11863770, 12210870, 32583580, 12216880,
    25870345, 14121591, 22050597, 14899505, 12963533, 19150597, 40003125, 13367990, 12349776,
    14358426, 29143263, 21212765, 41790281, 22402291, 47003407, 25089705, 34001516, 19295132,
    34317828, 27628082, 38470027, 30768181, 160049365, 79364237, 51816391, 73087805, 61960427,
    40865482, 56906621, 96277810, 51397125, 40291416, 26569200, 17598238, 71470374, 93929614,
    22626536, 21271031, 58635620, 26828362, 40773934, 21112099, 20519108, 18909710, 23332669,
    19252776, 40659856, 19713439, 20417058, 26818730, 19413335, 20962367, 24274574, 49178892,
    58268026, 24820590, 32869407, 18322271, 28170175, 25750532, 26692622, 35468428, 172460464,
    105599778, 26721409, 49508046, 44843508, 29229598, 31379598, 56091129, 59003136, 22720528,
    44629038, 71920482, 23476793, 24112336, 50401710, 44511171, 67809887, 34491383, 64675732,
    46125325, 51362558, 65089092, 108451156, 888065159,
];

/// The count of occurences of each byte, computed the books1, books3,
/// and github datasets from the ChatGPT paper, a snapshot of PyPi and
/// rust's crates.io, and tecnical manuals such as Intel's, RiscV, and APIC.
pub const ENGLISH_COUNTS: [usize; 256] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    49828,
    703173789,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    11493760988,
    18572189,
    267818439,
    28245367,
    5574283,
    3449544,
    3542054,
    274650523,
    56983715,
    57754995,
    91605704,
    1858336,
    791851173,
    110011697,
    810049509,
    18549998,
    79723340,
    120849889,
    76206506,
    48025583,
    41356746,
    42518860,
    34810597,
    33717511,
    37254082,
    53339565,
    54887498,
    39748489,
    1835316,
    3752624,
    4119661,
    53150279,
    507482,
    161845968,
    95283582,
    115594056,
    73249174,
    82802652,
    60656891,
    58356104,
    110134702,
    251093124,
    44142193,
    28610924,
    72027740,
    104524375,
    71818372,
    61631865,
    81170480,
    5962443,
    74143436,
    164126413,
    195693562,
    26938445,
    24038903,
    85739631,
    4027691,
    37616886,
    5613600,
    6689702,
    7683139,
    6738685,
    413527,
    127173912,
    4977458,
    4202285762,
    726407848,
    1419574362,
    2138564223,
    6460421408,
    1040236262,
    1053295461,
    2697729947,
    3530141643,
    66982024,
    445270848,
    2184394799,
    1248015262,
    3630112713,
    3909952380,
    969371374,
    68419504,
    3131600306,
    3271559193,
    4479812442,
    1534721601,
    513023117,
    952339956,
    98390076,
    916099587,
    61659941,
    2777580,
    14807048,
    2770077,
    415284,
    0,
    69395676,
    6467123,
    5451372,
    3730930,
    3649589,
    1863173,
    1450675,
    1923010,
    2311842,
    2081360,
    725004,
    2441735,
    1468587,
    1892071,
    1053004,
    1266096,
    1497178,
    869716,
    1119234,
    28405870,
    33828238,
    1219402,
    1194723,
    1229524,
    800461,
    1120976,
    929895,
    953631,
    1107303,
    711369,
    453636,
    1736038,
    7737586,
    6304501,
    4647578,
    1864755,
    14272925,
    6799294,
    1455925,
    2521930,
    6424451,
    33566358,
    3399623,
    4560143,
    1985137,
    9642242,
    2762142,
    2060971,
    4146659,
    5532271,
    2619077,
    8572840,
    2759580,
    3644308,
    5958347,
    1711800,
    4050524,
    3669791,
    3919341,
    6552963,
    6608514,
    7334501,
    3724030,
    4849238,
    0,
    0,
    14238656,
    102548910,
    3978239,
    2326408,
    54979,
    98377,
    356980,
    88874,
    122872,
    92699,
    46285,
    860,
    15312756,
    8056923,
    14112646,
    5844438,
    71274,
    24600,
    234,
    1366,
    9027,
    139961,
    919644,
    1707961,
    14705,
    91851,
    1891,
    19,
    1,
    320,
    7881691,
    4988878,
    68705452,
    4121145,
    1871060,
    3860075,
    2469440,
    1866169,
    1547218,
    1086813,
    179437,
    470262,
    598868,
    131084,
    45145,
    44481,
    3169,
    0,
    1,
    4,
    24,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

#[derive(Debug, Clone, MemDbg, MemSize, Default)]
/// Statistics of the encoded data.
pub struct Stats {
    /// Maximum shared prefix in bits.
    pub max_lcp: usize,
    /// The total sum of the shared prefix in bits.
    pub sum_lcp: usize,

    /// Maximum string length in bytes.
    pub max_str_len: usize,
    /// The total sum of the strings length in bytes.
    pub sum_str_len: usize,

    /// The number of bits used to store the rear lengths in data.
    pub code_bits: usize,
    /// The number of bits used to store the suffixes in data.
    pub suffixes_bits: usize,
}

/**

Immutable lists of strings compressed by prefix omission via rear coding.

Prefix omission compresses a list of strings omitting the common prefixes
of consecutive strings. To do so, it stores the length of what remains
after the common prefix (hence, rear coding). It is usually applied
to lists strings sorted in ascending order.

The encoding is done in blocks of `k` strings: in each block the first string is encoded
completely, wheres the other strings are encoded with the common prefix
removed. This is done to be able to have random accesses.

*/
#[derive(Debug, Clone, Epserde, MemDbg, MemSize)]
pub struct CompressedRearCodedList<DF: DecoderFactory = huffman::DecoderFactory> {
    /// The number of strings in a block; this value trades off compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// Whether the strings are sorted.
    is_sorted: bool,
    /// The encoded strings, `\0`-terminated.
    decoder_factory: DF,
}

#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct CompressedRearCodedListBuilder<E: Encoder = huffman::Encoder> {
    /// The number of strings in a block; this value trades off compression for speed.
    k: usize,
    /// Number of encoded strings.
    len: usize,
    /// Whether the strings are sorted.
    is_sorted: bool,
    /// Where the data is encoded.
    encoder: E,
    /// Statistics of the encoded data.
    stats: Stats,
    /// Cache of the last encoded string for incremental encoding.
    last_str: Vec<u8>,
}

impl<E: Encoder> CompressedRearCodedListBuilder<E> {
    #[inline]
    pub fn new(k: usize, encoder: E) -> Self {
        Self {
            encoder,
            last_str: Vec::with_capacity(1024),
            len: 0,
            is_sorted: true,
            k,
            stats: Default::default(),
        }
    }

    #[inline]
    pub fn build(self) -> Result<CompressedRearCodedList<E::DecoderFactory>> {
        Ok(CompressedRearCodedList {
            decoder_factory: self.encoder.into_decoder_factory()?,
            len: self.len,
            is_sorted: self.is_sorted,
            k: self.k,
        })
    }

    #[inline]
    /// Encode and append a string to the end of the list.
    pub fn push(&mut self, string: impl AsRef<str>) {
        let string = string.as_ref();
        // update stats
        self.stats.max_str_len = self.stats.max_str_len.max(string.len());
        self.stats.sum_str_len += string.len();

        let (lcp, order) = longest_common_prefix(&self.last_str, string.as_bytes());

        if order == core::cmp::Ordering::Greater {
            self.is_sorted = false;
        }

        // at every multiple of k we just encode the string as is
        let to_encode = if self.len % self.k == 0 {
            self.encoder.push_block_offset();
            // just encode the whole string
            string.as_bytes()
        } else {
            // update the stats
            self.stats.max_lcp = self.stats.max_lcp.max(lcp);
            self.stats.sum_lcp += lcp;
            // encode the len of the bytes in data
            let rear_length = self.last_str.len() - lcp;
            // update stats
            self.stats.code_bits += self.encoder.encode_length(rear_length);
            // return the delta suffix
            &string.as_bytes()[lcp..]
        };
        let bits = self.encoder.encode_bytes(to_encode);
        self.stats.suffixes_bits += bits;

        // put the string as last_str for the next iteration
        self.last_str.clear();
        self.last_str.extend_from_slice(string.as_bytes());
        self.len += 1;
    }

    #[inline]
    /// Append all the strings from an iterator to the end of the list
    pub fn extend<S: AsRef<str>, L: IntoLender>(&mut self, into_lender: L)
    where
        L::Lender: for<'lend> Lending<'lend, Lend = S>,
    {
        for_!(string in into_lender {
            self.push(string);
        });
    }

    /// Print in an human readable format the statistics of the RCL
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
}

impl<DF: DecoderFactory> CompressedRearCodedList<DF> {
    /// Write the index-th string to `result` as bytes. This is useful to avoid
    /// allocating a new string for every query and skipping the UTF-8 validity
    /// check.
    #[inline(always)]
    pub fn get_inplace(&self, index: usize, result: &mut Vec<u8>) {
        result.clear();
        let block_idx = index / self.k;
        let offset = index % self.k;

        let mut decoder = self.decoder_factory.get_decoder(block_idx).unwrap();

        // decode the first string in the block
        decoder.decode_bytes(result);

        for _ in 0..offset {
            // get how much data to throw away
            let len = decoder.decode_length();
            // throw away the data
            result.truncate(result.len() - len);
            decoder.decode_bytes(result);
        }
    }
}

impl<'a, 'all, DF: DecoderFactory> Lending<'all> for &'a CompressedRearCodedList<DF> {
    type Lend = &'all str;
}

impl<'a, DF: DecoderFactory> IntoLender for &'a CompressedRearCodedList<DF> {
    type Lender = Iterator<'a, DF>;
    #[inline(always)]
    fn into_lender(self) -> Iterator<'a, DF> {
        Iterator::new(self)
    }
}

impl<DF: DecoderFactory> IndexedDict for CompressedRearCodedList<DF> {
    type Output = String;
    type Input = str;

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output {
        let mut result = Vec::with_capacity(128);
        self.get_inplace(index, &mut result);
        String::from_utf8(result).unwrap()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

/// Sequential iterator over the strings.
#[derive(MemDbg, MemSize)]
pub struct Iterator<'a, DF: DecoderFactory> {
    rca: &'a CompressedRearCodedList<DF>,
    buffer: Vec<u8>,
    decoder: DF::Decoder<'a>,
    index: usize,
}

impl<'a, DF: DecoderFactory> core::fmt::Debug for Iterator<'a, DF>
where
    DF: core::fmt::Debug,
    DF::Decoder<'a>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Iterator")
            .field("rca", &self.rca)
            .field("buffer", &self.buffer)
            .field("decoder", &self.decoder)
            .field("index", &self.index)
            .finish()
    }
}

impl<'a, DF: DecoderFactory> Clone for Iterator<'a, DF>
where
    DF::Decoder<'a>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            rca: self.rca,
            buffer: self.buffer.clone(),
            decoder: self.decoder.clone(),
            index: self.index,
        }
    }
}

#[derive(MemDbg, MemSize)]
pub struct ValueIterator<'a, DF: DecoderFactory> {
    iter: Iterator<'a, DF>,
}

impl<'a, DF: DecoderFactory> core::fmt::Debug for ValueIterator<'a, DF>
where
    DF: core::fmt::Debug,
    DF::Decoder<'a>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueIterator")
            .field("iter", &self.iter)
            .finish()
    }
}

impl<'a, DF: DecoderFactory> Clone for ValueIterator<'a, DF>
where
    DF::Decoder<'a>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, DF: DecoderFactory> std::iter::Iterator for ValueIterator<'a, DF> {
    type Item = String;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|v| unsafe { String::from_utf8_unchecked(Vec::from(v)) })
    }
}

impl<'a, DF: DecoderFactory> Iterator<'a, DF> {
    pub fn new(rca: &'a CompressedRearCodedList<DF>) -> Self {
        Self {
            rca,
            buffer: Vec::with_capacity(128),
            decoder: rca.decoder_factory.get_decoder(0).unwrap(),
            index: 0,
        }
    }

    pub fn new_from(rca: &'a CompressedRearCodedList<DF>, start_index: usize) -> Self {
        let block = start_index / rca.k;
        let offset = start_index % rca.k;

        let decoder = rca.decoder_factory.get_decoder(block).unwrap();
        let mut res = Iterator {
            rca,
            index: block * rca.k,
            decoder,
            buffer: Vec::with_capacity(128),
        };
        for _ in 0..offset {
            res.next();
        }
        res
    }
}

impl<'a, 'b, DF: DecoderFactory> Lending<'a> for Iterator<'b, DF> {
    type Lend = &'a str;
}

impl<'a, DF: DecoderFactory> Lender for Iterator<'a, DF> {
    #[inline]
    /// A next that returns a reference to the inner buffer containg the string.
    /// This is useful to avoid allocating a new string for every query if you
    /// don't need to keep the string around.
    fn next(&mut self) -> Option<&'_ str> {
        if self.index >= self.rca.len() {
            return None;
        }

        if self.index % self.rca.k == 0 {
            // just copy the data
            self.buffer.clear();
            self.decoder.decode_bytes(&mut self.buffer);
        } else {
            let len = self.decoder.decode_length();
            self.buffer.truncate(self.buffer.len() - len);
            self.decoder.decode_bytes(&mut self.buffer);
        }
        self.index += 1;

        Some(unsafe { std::str::from_utf8_unchecked(&self.buffer) })
    }
}

impl<'a, DF: DecoderFactory> ExactSizeLender for Iterator<'a, DF> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.rca.len() - self.index
    }
}

impl<'a, DF: DecoderFactory> IntoIterator for &'a CompressedRearCodedList<DF> {
    type Item = String;
    type IntoIter = ValueIterator<'a, DF>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        ValueIterator {
            iter: Iterator::new(self),
        }
    }
}

impl<DF: DecoderFactory> CompressedRearCodedList<DF> {
    #[inline(always)]
    pub fn iter_from(&self, from: usize) -> ValueIterator<'_, DF> {
        ValueIterator {
            iter: Iterator::new_from(self, from),
        }
    }
}

#[inline(always)]
/// Compute the longest common prefix between two strings as bytes.
fn longest_common_prefix(a: &[u8], b: &[u8]) -> (usize, core::cmp::Ordering) {
    let min_len = a.len().min(b.len());
    // normal lcp computation
    let mut i = 0;
    while i < min_len && a[i] == b[i] {
        i += 1;
    }
    // TODO!: try to optimize with vpcmpeqb pextrb and leading count ones
    if i < min_len {
        (i, a[i].cmp(&b[i]))
    } else {
        (i, a.len().cmp(&b.len()))
    }
}

#[cfg(test)]
#[cfg_attr(test, test)]
fn test_longest_common_prefix() {
    let str1 = b"absolutely";
    let str2 = b"absorption";
    assert_eq!(
        longest_common_prefix(str1, str2),
        (4, core::cmp::Ordering::Less),
    );
    assert_eq!(
        longest_common_prefix(str1, str1),
        (str1.len(), core::cmp::Ordering::Equal)
    );
    assert_eq!(
        longest_common_prefix(str2, str2),
        (str2.len(), core::cmp::Ordering::Equal)
    );
}

#[cfg(test)]
fn read_into_lender<L: IntoLender>(into_lender: L) -> usize
where
    for<'a> <L::Lender as Lending<'a>>::Lend: AsRef<str>,
{
    let mut iter = into_lender.into_lender();
    let mut c = 0;
    while let Some(s) = iter.next() {
        c += s.as_ref().len();
    }

    c
}

#[cfg(test)]
#[cfg_attr(test, test)]
fn test_into_lend() {
    use dsi_bitstream::codes::huffman::HuffmanTree;

    let huffman = HuffmanTree::new(&DATA_COUNTS).unwrap();
    let mut builder = CompressedRearCodedListBuilder::new(
        4,
        huffman::Encoder::new(huffman, ef::OffsetsBuilder::default()),
    );
    builder.push("a");
    builder.push("b");
    builder.push("c");
    builder.push("d");
    let rcl = builder.build().unwrap();
    //read_into_lender::<&CompressedRearCodedList>(&rcl);
}
