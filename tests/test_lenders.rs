/*
 * SPDX-FileCopyrightText: 2025 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fmt::Debug;
use std::io::Cursor;
#[cfg(feature = "flate2")]
use std::io::Write;

use anyhow::{Context, Result, bail, ensure};
use fallible_iterator::{FallibleIterator, IntoFallibleIterator};
#[cfg(feature = "flate2")]
use flate2::write::GzEncoder;
use lender::{
    FallibleIteratorExt, FallibleLender, FallibleLending, IntoFallibleLender, IntoIteratorExt,
    IteratorExt, Lender, check_covariance_fallible, covar_mut, fallible_lend,
};

use sux::utils::lenders::*;

fn test_lender<
    L: FallibleRewindableLender<
            RewindError: Debug + std::error::Error + Send + Sync + 'static,
            Error: Debug + std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend (impl ?Sized + AsRef<str>)>,
>(
    mut lender: L,
) -> Result<()> {
    for pass in 0..5 {
        for i in 0..3 {
            match lender.next() {
                Ok(Some(got)) => {
                    let got = got.as_ref();
                    let expected = ["foo", "bar", "baz"][i];
                    ensure!(
                        got == expected,
                        "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                    );
                }
                Err(e) => bail!("Could not read item {i} of pass {pass}: {e:?}"),
                Ok(None) => bail!("Found only {i} items at pass {pass}"),
            }
        }
        if lender.next()?.is_some() {
            bail!("Found extra item after pass {pass}");
        }

        lender = lender.rewind()?;
    }

    Ok(())
}

#[test]
fn test_line_lender() -> Result<()> {
    let mut buf = Cursor::new(b"foo\nbar\nbaz\n");
    test_lender(LineLender::new(&mut buf))?;

    #[cfg(feature = "deko")]
    {
        use std::io::BufReader;

        buf.set_position(0);
        test_lender(DekoLineLender::new(&mut buf)?)?;
        buf.set_position(0);
        test_lender(DekoBufLineLender::new(BufReader::new(&mut buf))?)?;
    }
    Ok(())
}

#[cfg(feature = "zstd")]
#[test]
fn test_zstd_line_lender() -> Result<()> {
    let mut buf = Cursor::new(
        zstd::stream::encode_all(Cursor::new(b"foo\nbar\nbaz\n"), 1).context("Could not encode")?,
    );
    test_lender(ZstdLineLender::new(&mut buf).context("Could not initialize lender")?)?;

    #[cfg(feature = "deko")]
    {
        use std::io::BufReader;

        buf.set_position(0);
        test_lender(DekoLineLender::new(&mut buf)?)?;
        buf.set_position(0);
        test_lender(DekoBufLineLender::new(BufReader::new(&mut buf))?)?;
    }
    Ok(())
}

#[cfg(feature = "flate2")]
#[test]
fn test_gzip_line_lender() -> Result<()> {
    let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"foo\nbar\nbaz\n")?;
    let mut buf = Cursor::new(encoder.finish().context("Could not encode")?);

    test_lender(GzipLineLender::new(&mut buf).context("Could not initialize lender")?)?;

    #[cfg(feature = "deko")]
    {
        use std::io::BufReader;

        buf.set_position(0);
        test_lender(DekoLineLender::new(&mut buf)?)?;
        buf.set_position(0);
        test_lender(DekoBufLineLender::new(BufReader::new(&mut buf))?)?;
    }
    Ok(())
}

#[test]
fn test_from() -> Result<()> {
    test_lender(FromCloneableIntoIterator::from(["foo", "bar", "baz"]))?;
    test_lender(FromSlice::new(["foo", "bar", "baz"].as_slice()))?;

    // Test From trait implementation for FromSlice
    test_lender(FromSlice::from(["foo", "bar", "baz"].as_slice()))?;

    // Test FromIntoIterator with a Vec (where &Vec implements IntoIterator)
    let vec = vec!["foo", "bar", "baz"];
    test_lender(FromIntoIterator::new(&vec))?;
    // Test From trait for FromIntoIterator
    test_lender(FromIntoIterator::from(&vec))?;

    // Test FromIntoIterator with an array (where &[T] implements IntoIterator)
    let array = ["foo", "bar", "baz"];
    test_lender(FromIntoIterator::new(&array))?;
    // Test From trait for FromIntoIterator
    test_lender(FromIntoIterator::from(&array))?;

    Ok(())
}

// Test support for FromIntoFallibleLender
struct FallibleVecLender<T> {
    vec: Vec<T>,
    index: usize,
}

impl<'lend, T> FallibleLending<'lend> for FallibleVecLender<T> {
    type Lend = &'lend T;
}

impl<T> FallibleLender for FallibleVecLender<T> {
    check_covariance_fallible!();
    type Error = std::io::Error;
    fn next(&mut self) -> Result<Option<&T>, std::io::Error> {
        if self.index < self.vec.len() {
            let item = &self.vec[self.index];
            self.index += 1;
            Ok(Some(item))
        } else {
            Ok(None)
        }
    }
}

struct VecWrapper {
    vec: Vec<&'static str>,
}

impl IntoFallibleLender for &VecWrapper {
    type Error = std::io::Error;
    type FallibleLender = FallibleVecLender<&'static str>;
    fn into_fallible_lender(self) -> Self::FallibleLender {
        FallibleVecLender {
            vec: self.vec.clone(),
            index: 0,
        }
    }
}

#[test]
fn test_from_into_fallible_lender() -> Result<()> {
    let wrapper = VecWrapper {
        vec: vec!["foo", "bar", "baz"],
    };
    test_lender(FromIntoFallibleLender::new(&wrapper))?;
    // Test From trait for FromIntoFallibleLender
    test_lender(FromIntoFallibleLender::from(&wrapper))?;
    Ok(())
}

// Test support for FromIntoFallibleIterator
struct FallibleVecIter {
    items: Vec<&'static str>,
    index: usize,
}

impl FallibleIterator for FallibleVecIter {
    type Item = &'static str;
    type Error = std::io::Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.index < self.items.len() {
            let item = self.items[self.index];
            self.index += 1;
            Ok(Some(item))
        } else {
            Ok(None)
        }
    }
}

struct FallibleVec {
    items: Vec<&'static str>,
}

impl IntoFallibleIterator for &FallibleVec {
    type Item = &'static str;
    type Error = std::io::Error;
    type IntoFallibleIter = FallibleVecIter;

    fn into_fallible_iter(self) -> Self::IntoFallibleIter {
        FallibleVecIter {
            items: self.items.clone(),
            index: 0,
        }
    }
}

#[test]
fn test_from_into_fallible_iterator() -> Result<()> {
    let fallible_vec = FallibleVec {
        items: vec!["foo", "bar", "baz"],
    };
    test_lender(FromIntoFallibleIterator::new(&fallible_vec))?;
    // Test From trait for FromIntoFallibleIterator
    test_lender(FromIntoFallibleIterator::from(&fallible_vec))?;
    Ok(())
}

/*
#[test]
fn test_from_into_lender_factory() -> Result<()> {
    test_lender(
        FromIntoLenderFactory::new(|| -> Result<_, std::io::Error> {
            Ok(["foo", "bar", "baz"].into_into_lender())
        })
        .context("Could not initialize lender")?,
    )
}
*/

/*
#[test]
fn test_from_result_lender_factory() -> Result<()> {
    let items = || {
        [
            Ok("foo"),
            Ok("bar"),
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "error!")),
            Ok("baz"),
        ]
    };
    let mut lender = FromResultLenderFactory::new(|| -> Result<_, std::io::Error> {
        Ok(items().into_iter().into_lender())
    })
    .context("Could not initialize lender")?;

    let items = items();

    for pass in 0..5 {
        for i in 0..4 {
            match lender.next() {
                Some(got) => {
                    let got = got;
                    let expected = &items[i];
                    ensure!(
                        got.is_ok() == expected.is_ok(),
                        "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                    );
                    if got.is_ok() {
                        ensure!(
                            *got.as_ref().unwrap() == expected.as_ref().unwrap(),
                            "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                        );
                    }
                }
                None => bail!("Found only {i} items at pass {pass}"),
            }
        }
        if let Some(extra) = lender.next().map(Result::unwrap) {
            bail!("Found extra item after pass {pass}: {}", extra);
        }

        lender = lender.rewind().context("Could not rewind")?;
    }

    Ok(())
}
*/

#[test]
fn test_map() {
    let data = vec![1, 2, 3];

    let mut iter = FromSlice::new(&data).map(covar_mut!(for<'lend> |x: &'lend i32| -> Result<
        i32,
        std::convert::Infallible,
    > { Ok(x * 2) }));

    assert_eq!(iter.next().unwrap(), Some(2));
    assert_eq!(iter.next().unwrap(), Some(4));
    assert_eq!(iter.next().unwrap(), Some(6));
    assert_eq!(iter.next().unwrap(), None);
}

#[test]
fn test_flatten() {
    let data = vec![
        vec![1, 2, 3].into_iter().into_lender().into_fallible(),
        vec![1, 2, 3].into_iter().into_lender().into_fallible(),
        vec![1, 2, 3].into_iter().into_lender().into_fallible(),
    ];

    let mut lender = data.into_iter().into_lender().into_fallible().flatten();

    assert_eq!(lender.next().unwrap(), Some(1));
    assert_eq!(lender.next().unwrap(), Some(2));
    assert_eq!(lender.next().unwrap(), Some(3));
    assert_eq!(lender.next().unwrap(), Some(1));
    assert_eq!(lender.next().unwrap(), Some(2));
    assert_eq!(lender.next().unwrap(), Some(3));
    assert_eq!(lender.next().unwrap(), Some(1));
    assert_eq!(lender.next().unwrap(), Some(2));
    assert_eq!(lender.next().unwrap(), Some(3));
    assert_eq!(lender.next().unwrap(), None);
}

#[test]
fn test_line_lender_new() {
    let data = b"line1\nline2\n";
    let cursor = Cursor::new(&data[..]);
    let mut lender = LineLender::new(cursor);

    assert_eq!(lender.next().unwrap(), Some("line1"));
    assert_eq!(lender.next().unwrap(), Some("line2"));
    assert_eq!(lender.next().unwrap(), None);
}

#[test]
fn test_line_lender_rewind() {
    let data = b"first\nsecond\n";
    let cursor = Cursor::new(&data[..]);
    let mut lender = LineLender::new(cursor);

    assert_eq!(lender.next().unwrap(), Some("first"));
    assert_eq!(lender.next().unwrap(), Some("second"));

    let mut lender = lender.rewind().unwrap();

    assert_eq!(lender.next().unwrap(), Some("first"));
    assert_eq!(lender.next().unwrap(), Some("second"));
}

// Test FromSlice adapter
#[test]
fn test_from_slice_new() {
    let data = vec![1, 2, 3];
    let mut lender = FromSlice::new(data.as_slice());

    assert_eq!(lender.next().unwrap(), Some(&1));
    assert_eq!(lender.next().unwrap(), Some(&2));
    assert_eq!(lender.next().unwrap(), Some(&3));
    assert_eq!(lender.next().unwrap(), None);
}

#[test]
fn test_from_slice_rewind() {
    let data = vec![10, 20, 30];
    let mut lender = FromSlice::new(data.as_slice());

    assert_eq!(lender.next().unwrap(), Some(&10));
    assert_eq!(lender.next().unwrap(), Some(&20));

    let mut lender = lender.rewind().unwrap();

    assert_eq!(lender.next().unwrap(), Some(&10));
    assert_eq!(lender.next().unwrap(), Some(&20));
    assert_eq!(lender.next().unwrap(), Some(&30));
    assert_eq!(lender.next().unwrap(), None);
}

#[test]
fn test_from_slice_from_trait() {
    let data = [1, 2, 3];
    let mut lender = FromSlice::from(&data[..]);

    assert_eq!(lender.next().unwrap(), Some(&1));
    assert_eq!(lender.next().unwrap(), Some(&2));
}

// Test FromCloneableIntoIterator adapter
#[test]
fn test_from_cloneable_into_iterator_new() {
    let mut lender = FromCloneableIntoIterator::new(0..3);

    assert_eq!(lender.next().unwrap(), Some(&0));
    assert_eq!(lender.next().unwrap(), Some(&1));
    assert_eq!(lender.next().unwrap(), Some(&2));
    assert_eq!(lender.next().unwrap(), None);
}

#[test]
fn test_from_cloneable_into_iterator_rewind() {
    let mut lender = FromCloneableIntoIterator::new(5..8);

    assert_eq!(lender.next().unwrap(), Some(&5));
    assert_eq!(lender.next().unwrap(), Some(&6));

    let mut lender = lender.rewind().unwrap();

    assert_eq!(lender.next().unwrap(), Some(&5));
    assert_eq!(lender.next().unwrap(), Some(&6));
    assert_eq!(lender.next().unwrap(), Some(&7));
}

#[test]
fn test_from_cloneable_into_iterator_from_trait() {
    let mut lender = FromCloneableIntoIterator::from(1..4);

    assert_eq!(lender.next().unwrap(), Some(&1));
    assert_eq!(lender.next().unwrap(), Some(&2));
}

// Test FromIntoIterator adapter
#[test]
fn test_from_into_iterator_new() {
    let data = vec![10, 20, 30];
    let mut lender = FromIntoIterator::new(&data);

    assert_eq!(lender.next().unwrap(), Some(&&10));
    assert_eq!(lender.next().unwrap(), Some(&&20));
    assert_eq!(lender.next().unwrap(), Some(&&30));
    assert_eq!(lender.next().unwrap(), None);
}

#[test]
fn test_from_into_iterator_rewind() {
    let data = vec![100, 200];
    let mut lender = FromIntoIterator::new(&data);

    assert_eq!(lender.next().unwrap(), Some(&&100));
    assert_eq!(lender.next().unwrap(), Some(&&200));

    let mut lender = lender.rewind().unwrap();

    assert_eq!(lender.next().unwrap(), Some(&&100));
    assert_eq!(lender.next().unwrap(), Some(&&200));
}

#[test]
fn test_from_into_iterator_from_trait() {
    let data = vec![1, 2, 3];
    let mut lender = FromIntoIterator::from(&data);

    assert_eq!(lender.next().unwrap(), Some(&&1));
}

// Test FromIntoLenderFactory adapter
#[test]
fn test_from_into_lender_factory_new() {
    use lender::FromIntoIter;

    let mut count = 0;
    let mut lender = FromIntoLenderFactory::new(|| {
        count += 1;
        Ok::<FromIntoIter<std::ops::Range<i32>>, core::convert::Infallible>(
            (0..count).into_into_lender(),
        )
    })
    .unwrap();

    assert_eq!(lender.next().unwrap(), Some(0));
    assert_eq!(lender.next().unwrap(), None);

    let mut lender = lender.rewind().unwrap();
    assert_eq!(lender.next().unwrap(), Some(0));
    assert_eq!(lender.next().unwrap(), Some(1));
    assert_eq!(lender.next().unwrap(), None);
}

// Test FromIntoFallibleLenderFactory adapter
#[test]
fn test_from_into_fallible_lender_factory_new() {
    use fallible_iterator::IteratorExt as FallibleIteratorExt;
    use lender::FromFallibleIter;

    let mut count = 0;
    let mut lender = FromIntoFallibleLenderFactory::new(|| {
        count += 1;
        Ok::<
            FromFallibleIter<fallible_iterator::IntoFallible<std::ops::Range<i32>>>,
            core::convert::Infallible,
        >(
            (0..count)
                .into_iter()
                .into_fallible()
                .into_fallible_lender(),
        )
    })
    .unwrap();

    assert_eq!(lender.next().unwrap(), Some(0));
    assert_eq!(lender.next().unwrap(), None);

    let mut lender = lender.rewind().unwrap();
    assert_eq!(lender.next().unwrap(), Some(0));
    assert_eq!(lender.next().unwrap(), Some(1));
    assert_eq!(lender.next().unwrap(), None);
}

// Test lender adapter implementations
#[test]
fn test_enumerate_rewind() {
    let data = vec![10, 20, 30];
    let lender = FromSlice::new(data.as_slice());
    let mut enumerated = lender.enumerate();

    assert_eq!(enumerated.next().unwrap(), Some((0, &10)));
    assert_eq!(enumerated.next().unwrap(), Some((1, &20)));

    let mut enumerated = enumerated.rewind().unwrap();

    assert_eq!(enumerated.next().unwrap(), Some((0, &10)));
    assert_eq!(enumerated.next().unwrap(), Some((1, &20)));
}

#[test]
fn test_fuse_rewind() {
    let data = vec![1, 2];
    let lender = FromSlice::new(data.as_slice());
    let mut fused = lender.fuse();

    assert_eq!(fused.next().unwrap(), Some(&1));
    assert_eq!(fused.next().unwrap(), Some(&2));
    assert_eq!(fused.next().unwrap(), None);

    let mut fused = fused.rewind().unwrap();

    assert_eq!(fused.next().unwrap(), Some(&1));
}

#[test]
fn test_take_rewind() {
    let data = vec![1, 2, 3, 4, 5];
    let lender = FromSlice::new(data.as_slice());
    let mut taken = lender.take(4);

    // Take first two, leaving take count at 2
    assert_eq!(taken.next().unwrap(), Some(&1));
    assert_eq!(taken.next().unwrap(), Some(&2));

    // Rewind - note: into_parts returns remaining count (2), not original (4)
    let mut taken = taken.rewind().unwrap();

    // After rewind with remaining count=2, we can only take 2 more
    assert_eq!(taken.next().unwrap(), Some(&1));
    assert_eq!(taken.next().unwrap(), Some(&2));
    assert_eq!(taken.next().unwrap(), None);
}

#[test]
fn test_skip_rewind() {
    let data = vec![1, 2, 3, 4, 5];
    let lender = FromSlice::new(data.as_slice());
    let mut skipped = lender.skip(2);

    // After skip, we start at element 3
    assert_eq!(skipped.next().unwrap(), Some(&3));
    assert_eq!(skipped.next().unwrap(), Some(&4));

    // Rewind - note: into_parts returns remaining skip count (0), not original (2)
    let mut skipped = skipped.rewind().unwrap();

    // After rewind with skip=0, we start from element 1
    assert_eq!(skipped.next().unwrap(), Some(&1));
    assert_eq!(skipped.next().unwrap(), Some(&2));
}

#[test]
fn test_step_by_rewind() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let lender = FromSlice::new(data.as_slice());
    let mut stepped = lender.step_by(2);

    assert_eq!(stepped.next().unwrap(), Some(&1));
    assert_eq!(stepped.next().unwrap(), Some(&3));

    let mut stepped = stepped.rewind().unwrap();

    assert_eq!(stepped.next().unwrap(), Some(&1));
    assert_eq!(stepped.next().unwrap(), Some(&3));
}

#[test]
fn test_chain_rewind() {
    let data1 = vec![1, 2];
    let data2 = vec![3, 4];
    let lender1 = FromSlice::new(data1.as_slice());
    let lender2 = FromSlice::new(data2.as_slice());
    let mut chained = lender1.chain(lender2);

    assert_eq!(chained.next().unwrap(), Some(&1));
    assert_eq!(chained.next().unwrap(), Some(&2));
    assert_eq!(chained.next().unwrap(), Some(&3));

    let mut chained = chained.rewind().unwrap();

    assert_eq!(chained.next().unwrap(), Some(&1));
    assert_eq!(chained.next().unwrap(), Some(&2));
}

#[test]
fn test_peekable_rewind() {
    let data = vec![1, 2, 3];
    let lender = FromSlice::new(data.as_slice());
    let mut peekable = lender.peekable();

    assert_eq!(peekable.next().unwrap(), Some(&1));
    assert_eq!(peekable.next().unwrap(), Some(&2));

    let mut peekable = peekable.rewind().unwrap();

    assert_eq!(peekable.next().unwrap(), Some(&1));
}

// Test intersperse rewind
#[test]
fn test_intersperse_rewind() {
    let data = vec![1, 2, 3];
    let lender = FromSlice::new(data.as_slice());
    let mut interspersed = lender.intersperse(&0);

    // Items: 1, 0, 2, 0, 3
    assert_eq!(interspersed.next().unwrap(), Some(&1));
    assert_eq!(interspersed.next().unwrap(), Some(&0));
    assert_eq!(interspersed.next().unwrap(), Some(&2));

    let mut interspersed = interspersed.rewind().unwrap();

    assert_eq!(interspersed.next().unwrap(), Some(&1));
    assert_eq!(interspersed.next().unwrap(), Some(&0));
}

#[test]
fn test_cycle_rewind() {
    let data = vec![1, 2];
    let lender = FromSlice::new(data.as_slice());
    let mut cycled = lender.cycle();

    assert_eq!(cycled.next().unwrap(), Some(&1));
    assert_eq!(cycled.next().unwrap(), Some(&2));
    assert_eq!(cycled.next().unwrap(), Some(&1));
    assert_eq!(cycled.next().unwrap(), Some(&2));

    let mut cycled = cycled.rewind().unwrap();

    assert_eq!(cycled.next().unwrap(), Some(&1));
    assert_eq!(cycled.next().unwrap(), Some(&2));
    assert_eq!(cycled.next().unwrap(), Some(&1));
}

#[test]
fn test_fallible_empty_rewind() {
    // Use FromSlice's lending type which satisfies FallibleLending
    let mut empty =
        lender::fallible_empty::<fallible_lend!(&'lend i32), core::convert::Infallible>();

    assert_eq!(FallibleLender::next(&mut empty).unwrap(), None);

    let mut empty = empty.rewind().unwrap();

    assert_eq!(FallibleLender::next(&mut empty).unwrap(), None);
}

#[test]
fn test_fallible_repeat_rewind() {
    let data = vec![42];
    let mut lender = FromSlice::new(data.as_slice());
    // Get one lend, then use it as a repeat source
    let value = FallibleLender::next(&mut lender).unwrap().unwrap();
    let mut repeated =
        lender::fallible_repeat::<fallible_lend!(&'lend i32), core::convert::Infallible>(value);

    assert_eq!(FallibleLender::next(&mut repeated).unwrap(), Some(&42));
    assert_eq!(FallibleLender::next(&mut repeated).unwrap(), Some(&42));

    let mut repeated = repeated.rewind().unwrap();

    assert_eq!(FallibleLender::next(&mut repeated).unwrap(), Some(&42));
    assert_eq!(FallibleLender::next(&mut repeated).unwrap(), Some(&42));
}

#[test]
fn test_map_rewind() {
    let data = vec![1, 2, 3];
    let lender = FromSlice::new(data.as_slice());
    let mut mapped = lender.map(covar_mut!(for<'lend> |x: &'lend i32| -> Result<
        i32,
        std::convert::Infallible,
    > { Ok(x * 10) }));

    assert_eq!(mapped.next().unwrap(), Some(10));
    assert_eq!(mapped.next().unwrap(), Some(20));

    let mut mapped = mapped.rewind().unwrap();

    assert_eq!(mapped.next().unwrap(), Some(10));
    assert_eq!(mapped.next().unwrap(), Some(20));
    assert_eq!(mapped.next().unwrap(), Some(30));
}
