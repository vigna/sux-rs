use serde::{ser, Serialize};
use std::io::Write;

pub struct Serializer<F: Write> {
    // Stream where we serialize the values
    file: F,
    bytes_written: usize,
}

#[derive(Debug)]
pub struct Error(String);

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self(format!("{}", value))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T>(msg: T) -> Self
    where
        T: std::fmt::Display,
    {
        Self(format!("{}", msg))
    }
}

impl<F: Write> Serializer<F> {
    pub fn new(file: F) -> Self {
        Self {
            file,
            bytes_written: 0,
        }
    }
}

impl<'a, F: Write> ser::Serializer for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn is_human_readable(&self) -> bool {
        false
    }

    fn serialize_bool(self, v: bool) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&[v as u8])?;
        self.bytes_written += 1;
        Ok(())
    }

    fn serialize_u8(self, v: u8) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 1;
        Ok(())
    }

    fn serialize_i8(self, v: i8) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 1;
        Ok(())
    }

    fn serialize_u16(self, v: u16) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 2;
        Ok(())
    }

    fn serialize_i16(self, v: i16) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 2;
        Ok(())
    }

    fn serialize_u32(self, v: u32) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }

    fn serialize_i32(self, v: i32) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }

    fn serialize_u64(self, v: u64) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 8;
        Ok(())
    }

    fn serialize_i64(self, v: i64) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 8;
        Ok(())
    }

    fn serialize_u128(self, v: u128) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 16;
        Ok(())
    }

    fn serialize_i128(self, v: i128) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 16;
        Ok(())
    }

    fn serialize_f32(self, v: f32) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }

    fn serialize_f64(self, v: f64) -> std::result::Result<Self::Ok, Self::Error> {
        self.file.write(&v.to_ne_bytes())?;
        self.bytes_written += 8;
        Ok(())
    }

    fn serialize_char(self, v: char) -> std::result::Result<Self::Ok, Self::Error> {
        // https://doc.rust-lang.org/std/primitive.char.html#method.from_u32
        self.file.write(&(v as u32).to_ne_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }

    fn serialize_str(self, v: &str) -> std::result::Result<Self::Ok, Self::Error> {
        // TODO!: len?
        self.file.write(v.as_bytes())?;
        self.bytes_written += v.as_bytes().len();
        Ok(())
    }

    fn serialize_bytes(self, v: &[u8]) -> std::result::Result<Self::Ok, Self::Error> {
        // TODO!: len?
        self.file.write(v)?;
        self.bytes_written += v.len();
        Ok(())
    }

    fn serialize_unit(self) -> std::result::Result<Self::Ok, Self::Error> {
        // no need?
        Ok(())
    }

    fn serialize_unit_struct(
        self,
        _name: &'static str,
    ) -> std::result::Result<Self::Ok, Self::Error> {
        // no need?
        Ok(())
    }

    fn serialize_none(self) -> std::result::Result<Self::Ok, Self::Error> {
        self.serialize_bool(false)
    }

    fn serialize_some<T: ?Sized>(self, value: &T) -> std::result::Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        self.serialize_bool(true)?;
        value.serialize(self)
    }

    fn serialize_tuple(
        self,
        _len: usize,
    ) -> std::result::Result<Self::SerializeTuple, Self::Error> {
        Ok(self)
    }

    fn serialize_seq(
        self,
        len: Option<usize>,
    ) -> std::result::Result<Self::SerializeSeq, Self::Error> {
        // we can only support sequences with known length
        // otherwise we would need to create a buffer to collect the iter
        // and then store the length anyway
        self.serialize_u64(len.unwrap() as u64)?;
        Ok(self)
    }

    fn serialize_map(
        self,
        len: Option<usize>,
    ) -> std::result::Result<Self::SerializeMap, Self::Error> {
        // we can only support sequences with known length
        // otherwise we would need to create a buffer to collect the iter
        // and then store the length anyway
        self.serialize_u64(len.unwrap() as u64)?;
        Ok(self)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> std::result::Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> std::result::Result<Self::SerializeTupleVariant, Self::Error> {
        self.serialize_u32(variant_index)?;
        Ok(self)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> std::result::Result<Self::SerializeStruct, Self::Error> {
        Ok(self)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
    ) -> std::result::Result<Self::Ok, Self::Error> {
        self.serialize_u32(variant_index)?;
        Ok(())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> std::result::Result<Self::SerializeStructVariant, Self::Error> {
        self.serialize_u32(variant_index)?;
        Ok(self)
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> std::result::Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> std::result::Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        self.serialize_u32(variant_index)?;
        value.serialize(self)
    }
}

impl<'a, F: Write> ser::SerializeSeq for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized>(&mut self, value: &T) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, F: Write> ser::SerializeTuple for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized>(&mut self, value: &T) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, F: Write> ser::SerializeMap for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized>(&mut self, key: &T) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        key.serialize(&mut **self)
    }

    fn serialize_value<T: ?Sized>(&mut self, value: &T) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, F: Write> ser::SerializeStruct for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, F: Write> ser::SerializeStructVariant for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, F: Write> ser::SerializeTupleStruct for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized>(&mut self, value: &T) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl<'a, F: Write> ser::SerializeTupleVariant for &'a mut Serializer<F> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized>(&mut self, value: &T) -> std::result::Result<(), Self::Error>
    where
        T: Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> std::result::Result<Self::Ok, Self::Error> {
        Ok(())
    }
}
