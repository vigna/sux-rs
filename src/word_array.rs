use crate::prelude::*;
use anyhow::{Context, Result};
use common_traits::*;
use log::info;
use mmap_rs::{Mmap, MmapMut};
use std::path::Path;

/// A simple wrapper around a slice of bytes interpreted as native-endianess words
/// with utility methods for mmapping.
pub struct WordArray<W: Word, B: AsRef<[u8]>> {
    data: B,
    _marker: core::marker::PhantomData<W>,
}

impl<W: Word, B: AsRef<[u8]>> WordArray<W, B> {
    #[inline(always)]
    pub fn get(&self, index: usize) -> W {
        if index >= self.len() {
            panic!("Index out of bounds: {} >= {}", index, self.len())
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> W {
        debug_assert!(index < self.len());
        *(self.data.as_ref().as_ptr() as *const W).add(index)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.as_ref().len() / W::BYTES
    }
}

impl<W: Word, B: AsRef<[u8]> + AsMut<[u8]>> WordArray<W, B> {
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: W) {
        if index >= self.len() {
            panic!("Index out of bounds: {} >= {}", index, self.len())
        } else {
            unsafe { self.set_unchecked(index, value) }
        }
    }
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: W) {
        debug_assert!(index < self.len());
        *(self.data.as_ref().as_ptr() as *mut W).add(index) = value;
    }
}

impl<W: Word> WordArray<W, Mmap> {
    /// Load a `.order` file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file_len = path.metadata()?.len();
        let file = std::fs::File::open(path)?;
        let data = unsafe {
            mmap_rs::MmapOptions::new(file_len as _)?
                .with_flags((crate::prelude::Flags::TRANSPARENT_HUGE_PAGES).mmap_flags())
                .with_file(file, 0)
                .map()?
        };
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(data.as_ptr() as *mut _, data.len(), libc::MADV_RANDOM)
        };
        Ok(Self {
            data,
            _marker: Default::default(),
        })
    }
}

impl<W: Word> WordArray<W, MmapMut> {
    /// Create a new `.order` file
    pub fn new_file<P: AsRef<Path>>(path: P, num_nodes: u64) -> Result<Self> {
        let path = path.as_ref();
        // compute the size of the file we are creating in bytes
        let file_len = num_nodes * core::mem::size_of::<u64>() as u64;
        info!(
            "The file {} will be {} bytes long.",
            path.to_string_lossy(),
            file_len
        );

        // create the file
        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .with_context(|| {
                format!("While creating the .order file: {}", path.to_string_lossy())
            })?;

        // fallocate the file with zeros so we can fill it without ever resizing it
        file.set_len(file_len)
            .with_context(|| "While fallocating the file with zeros")?;

        // create a mutable mmap to the file so we can directly write it in place
        let mmap = unsafe {
            mmap_rs::MmapOptions::new(file_len as _)?
                .with_file(file, 0)
                .map_mut()
                .with_context(|| "While mmapping the file")?
        };

        Ok(Self {
            data: mmap,
            _marker: Default::default(),
        })
    }

    /// Load a mutable `.order` file
    pub fn load_mut<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file_len = path.metadata()?.len();
        let file = std::fs::File::open(path)?;
        let data = unsafe {
            mmap_rs::MmapOptions::new(file_len as _)?
                .with_flags((crate::prelude::Flags::TRANSPARENT_HUGE_PAGES).mmap_flags())
                .with_file(file, 0)
                .map_mut()?
        };
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(data.as_ptr() as *mut _, data.len(), libc::MADV_RANDOM)
        };
        Ok(Self {
            data,
            _marker: Default::default(),
        })
    }
}
