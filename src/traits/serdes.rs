use crate::utils::*;
use anyhow::Result;
use std::{
    io::{Read, Seek, Write},
    mem::MaybeUninit,
    ops::Deref,
    path::Path,
    ptr::addr_of_mut,
    sync::Arc,
};

/// Possible backends of a [`MemCase`]. The `None` variant is used when the data structure is
/// created in memory; the `Memory` variant is used when the data structure is deserialized
/// from a file loaded into allocated memory; the `Mmap` variant is used when
/// the data structure is deserialized from a memory-mapped file.
pub enum MemBackend {
    /// No backend. The data structure is a standard Rust data structure.
    None,
    /// The backend is an allocated memory region.
    Memory(Vec<u64>),
    /// The backend is a memory-mapped file.
    Mmap(mmap_rs::Mmap),
}

/// Possible backends of a [`RefCase`]. See ['MemBackend'] for details.
#[derive(Clone)]
pub enum RefBackend {
    /// No backend. The data structure is a standard Rust data structure.
    None,
    /// The backend is an allocated memory region.
    Memory(Arc<Vec<u64>>),
    /// The backend is a memory-mapped file.
    Mmap(Arc<mmap_rs::Mmap>),
}

/// A wrapper keeping together a data structure and the memory
/// it was deserialized from. It is specifically designed for
/// the case of memory-mapped files, where the mapping must
/// be kept alive for the whole lifetime of the data structure.
/// It can also be used with data structures deserialized from
/// memory, although in that case it is not strictly necessary
/// (cloning each field would work); nonetheless, reading a
/// single block of memory with [`Read::read_exact`] can be
/// very fast, and using [`load`] is a way to ensure that
/// no cloning is performed.
///
/// [`MemCase`] instances can not be cloned, but references
/// to such instances can be shared freely.
///
/// [`MemCase`] implements [`Deref`] and [`AsRef`] to the
/// wrapped type, so it can be used almost transparently. However,
/// if you need to use a memory-mapped structure as a field in
/// a struct and you want to avoid `dyn`, you will have
/// to use [`MemCase`] as the type of the field.
/// [`MemCase`] implements [`From`] for the
/// wrapped type, using the no-op [None](`MemBackend::None`) variant
/// of [`MemBackend`], so a data structure can be [encased](encase_mem)
/// almost transparently.

pub struct MemCase<S>(S, MemBackend);

unsafe impl<S: Send> Send for MemCase<S> {}
unsafe impl<S: Sync> Sync for MemCase<S> {}

impl<S> Deref for MemCase<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, S> AsRef<S> for MemCase<S> {
    fn as_ref(&self) -> &S {
        &self.0
    }
}

/// Encases a data structure in a [`MemCase`] with no backend.
pub fn encase_mem<S>(s: S) -> MemCase<S> {
    MemCase(s, MemBackend::None)
}

impl<S: Send + Sync> From<S> for MemCase<S> {
    fn from(s: S) -> Self {
        encase_mem(s)
    }
}

/// A wrapper keeping together a reference (usually, to a slice)
/// and the memory that supports it. It is specifically designed for
/// the case of memory-mapped files, where the mapping must
/// be kept alive for the whole lifetime of the reference.
///
/// [`RefCase`] instances can be freely cloned, as they keeps an [`Arc`]
/// to the memory they reference.
///
/// [`RefCase`] implements [`Deref`] and [`AsRef`] to the
/// wrapped type, so it can be used almost transparently. However,
/// if you need to use a memory-mapped structure as a field in
/// a struct and you want to avoid `dyn`, you will have
/// to use [`RefCase`] as the type of the field.
/// [`RefCase`] implements [`From`] for the
/// wrapped type, using the no-op [None](`RefBackend::None`) variant
/// of [`RefBackend`], so a reference can be [encased](encase_ref)
/// almost transparently.

pub struct RefCase<'a, S: ?Sized>(&'a S, RefBackend);

impl<'a, S: ?Sized> Clone for RefCase<'a, S> {
    fn clone(&self) -> Self {
        Self(self.0, self.1.clone())
    }
}

unsafe impl<'a, S: Send + ?Sized> Send for RefCase<'a, S> {}
unsafe impl<'a, S: Sync + ?Sized> Sync for RefCase<'a, S> {}

impl<'a, S: ?Sized> Deref for RefCase<'a, S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, S: ?Sized> AsRef<S> for RefCase<'a, S> {
    fn as_ref(&self) -> &S {
        self.0
    }
}

/// Encases a data structure in a [`RefCase`] with no backend.
pub fn encase_ref<S>(s: &S) -> RefCase<S> {
    RefCase(s, RefBackend::None)
}

impl<'a, S: Send + Sync> From<&'a S> for RefCase<'a, S> {
    fn from(s: &'a S) -> Self {
        encase_ref(s)
    }
}

/// Mamory map a file and deserialize a data structure from it,
/// returning a [`MemCase`] containing the data structure and the
/// memory mapping.
#[allow(clippy::uninit_vec)]
pub fn map<'a, P: AsRef<Path>, S: Deserialize<'a>>(path: P) -> Result<MemCase<S>> {
    let file_len = path.as_ref().metadata()?.len();
    let file = std::fs::File::open(path)?;

    Ok({
        let mut uninit: MaybeUninit<MemCase<S>> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        let mmap = unsafe {
            mmap_rs::MmapOptions::new(file_len as _)?
                .with_file(file, 0)
                .map()?
        };

        unsafe {
            addr_of_mut!((*ptr).1).write(MemBackend::Mmap(mmap));
        }

        if let MemBackend::Mmap(mmap) = unsafe { &(*ptr).1 } {
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

/// Load a file into memory and deserialize a data structure from it,
/// returning a [`MemCase`] containing the data structure and the
/// memory. Excess bytes are zeroed out.
#[allow(clippy::uninit_vec)]
pub fn load<'a, P: AsRef<Path>, S: Deserialize<'a>>(path: P) -> Result<MemCase<S>> {
    let file_len = path.as_ref().metadata()?.len() as usize;
    let mut file = std::fs::File::open(path)?;
    let capacity = (file_len + 7) / 8;
    let mut mem = Vec::<u64>::with_capacity(capacity);
    unsafe {
        // This is safe because we are filling the vector
        // reading from a file.
        mem.set_len(capacity);
    }
    Ok({
        let mut uninit: MaybeUninit<MemCase<S>> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        let bytes: &mut [u8] = bytemuck::cast_slice_mut::<u64, u8>(mem.as_mut_slice());
        file.read_exact(&mut bytes[..file_len])?;
        // Fixes the last few bytes to guarantee zero-extension semantics
        // for bit vectors.
        bytes[file_len..].fill(0);

        unsafe {
            addr_of_mut!((*ptr).1).write(MemBackend::Memory(mem));
        }

        if let MemBackend::Memory(mem) = unsafe { &mut (*ptr).1 } {
            let (s, _) = S::deserialize(bytemuck::cast_slice::<u64, u8>(mem))?;

            unsafe {
                addr_of_mut!((*ptr).0).write(s);
            }

            unsafe { uninit.assume_init() }
        } else {
            unreachable!()
        }
    })
}

/// Mamory map a file, returning a [`MemCase`] containing a reference
/// to a slice of the given type filled with the file content.
/// Excess bytes are zeroed out.
pub fn map_slice<'a, P: AsRef<Path>, T: bytemuck::Pod>(path: P) -> Result<RefCase<'a, [T]>> {
    let file_len = path.as_ref().metadata()?.len();
    let file = std::fs::File::open(path)?;

    Ok({
        let mut uninit: MaybeUninit<RefCase<[T]>> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        let mmap = unsafe {
            mmap_rs::MmapOptions::new(file_len as _)?
                .with_file(file, 0)
                .map()?
        };

        unsafe {
            addr_of_mut!((*ptr).1).write(RefBackend::Mmap(Arc::new(mmap)));
        }

        if let RefBackend::Mmap(mmap) = unsafe { &(*ptr).1 } {
            let s = bytemuck::cast_slice::<u8, T>(mmap);
            unsafe {
                addr_of_mut!((*ptr).0).write(s);
            }

            unsafe { uninit.assume_init() }
        } else {
            unreachable!()
        }
    })
}

/// Loads a file in memory, returning a [`MemCase`] containing a reference
/// to a slice of the given type filled with the file content. Excess bytes
/// are zeroed out.
#[allow(clippy::uninit_vec)]
pub fn load_slice<'a, P: AsRef<Path>, T: bytemuck::Pod>(path: P) -> Result<RefCase<'a, [T]>> {
    let file_len = path.as_ref().metadata()?.len() as usize;
    let mut file = std::fs::File::open(path)?;
    let capacity = (file_len + 7) / 8;
    let mut mem = Vec::<u64>::with_capacity(capacity);
    unsafe {
        // This is safe because we are filling the vector
        // reading from a file.
        mem.set_len(capacity);
    }
    Ok({
        let mut uninit: MaybeUninit<RefCase<'a, [T]>> = MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr();

        let bytes: &mut [u8] = bytemuck::cast_slice_mut::<u64, u8>(mem.as_mut_slice());
        file.read_exact(&mut bytes[..file_len])?;
        // Fixes the last few bytes to guarantee zero-extension semantics
        // for bit vectors.
        bytes[file_len..].fill(0);

        unsafe {
            addr_of_mut!((*ptr).1).write(RefBackend::Memory(Arc::new(mem)));
        }

        if let RefBackend::Memory(mem) = unsafe { &mut (*ptr).1 } {
            let s: &[T] = bytemuck::cast_slice::<u64, T>(mem);

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
