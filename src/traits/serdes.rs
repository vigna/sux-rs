use crate::utils::*;
use anyhow::Result;
use std::{
    io::{Read, Seek, Write},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    path::Path,
    ptr::addr_of_mut,
};

use super::VSlice;

enum Backend {
    Mmap(mmap_rs::Mmap),
    Memory(Vec<u64>),
}
/// Encases a data structure together with its backend.
pub struct Encase<S>(S, Backend);

unsafe impl<S: Send> Send for Encase<S> {}
unsafe impl<S: Sync> Sync for Encase<S> {}

impl<S> Deref for Encase<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S> DerefMut for Encase<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S: VSlice> VSlice for Encase<S> {
    fn bit_width(&self) -> usize {
        self.0.bit_width()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        self.0.get_unchecked(index)
    }
}

pub fn map<'a, P: AsRef<Path>, S: Deserialize<'a>>(path: P) -> Result<Encase<S>> {
    let file_len = path.as_ref().metadata()?.len();
    let file = std::fs::File::open(path)?;

    Ok({
        let mut uninit: MaybeUninit<Encase<S>> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        let mmap = unsafe {
            mmap_rs::MmapOptions::new(file_len as _)?
                .with_file(file, 0)
                .map()?
        };

        unsafe {
            addr_of_mut!((*ptr).1).write(Backend::Mmap(mmap));
        }

        if let Backend::Mmap(mmap) = unsafe { &(*ptr).1 } {
            let (s, _) = S::deserialize(mmap)?;
            unsafe {
                addr_of_mut!((*ptr).0).write(s);
            }

            unsafe { uninit.assume_init() }
        } else {
            unreachable!()
        }
    })
}

pub fn load<'a, P: AsRef<Path>, S: Deserialize<'a>>(path: P) -> Result<Encase<S>> {
    let file_len = path.as_ref().metadata()?.len();
    let mut file = std::fs::File::open(path)?;
    let capacity = file_len as usize + 7 / 8;
    let mut mem = Vec::<u64>::with_capacity(capacity);
    unsafe {
        mem.set_len(capacity);
    }
    Ok({
        let mut uninit: MaybeUninit<Encase<S>> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        unsafe {
            addr_of_mut!((*ptr).1).write(Backend::Memory(mem));
        }

        if let Backend::Memory(mem) = unsafe { &mut (*ptr).1 } {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut::<u64, u8>(mem);
            file.read(&mut bytes[..capacity])?;

            let (s, _) = S::deserialize(bytes)?;

            unsafe {
                addr_of_mut!((*ptr).0).write(s);
            }

            unsafe { uninit.assume_init() }
        } else {
            unreachable!()
        }
    })
}

pub trait Serialize {
    fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize>;
}

pub trait Deserialize<'a>: Sized {
    /// a function that return a deserialzied values that might contains
    /// references to the backend
    fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])>;
}

macro_rules! impl_stuff{
    ($($ty:ty),*) => {$(

impl Serialize for $ty {
    #[inline(always)]
    fn serialize<F: Write>(&self, backend: &mut F) -> Result<usize> {
        Ok(backend.write(&self.to_ne_bytes())?)
    }
}

impl<'a> Deserialize<'a> for $ty {
    #[inline(always)]
    fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
        Ok((
            <$ty>::from_ne_bytes(backend[..core::mem::size_of::<$ty>()].try_into().unwrap()),
            &backend[core::mem::size_of::<$ty>()..],
        ))
    }
}
        impl<'a> Deserialize<'a> for &'a [$ty] {
            fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
                let (len, backend) = usize::deserialize(backend)?;
                let bytes = len * core::mem::size_of::<$ty>();
                let (_pre, data, after) = unsafe { backend[..bytes].align_to() };
                // TODO make error / we added padding so it's ok
                assert!(after.is_empty());
                Ok((data, &backend[bytes..]))
            }
        }
    )*};
}

impl_stuff!(usize, u8, u16, u32, u64);

impl<T: Serialize> Serialize for Vec<T> {
    fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
        let len = self.len();
        let mut bytes = 0;
        bytes += backend.write(&len.to_ne_bytes())?;
        // ensure alignement
        let file_pos = backend.stream_position()? as usize;
        for _ in 0..pad_align_to(file_pos, core::mem::size_of::<T>()) {
            bytes += backend.write(&[0])?;
        }
        // write the values
        for item in self {
            bytes += item.serialize(backend)?;
        }
        Ok(bytes)
    }
}
