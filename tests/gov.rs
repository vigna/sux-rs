use anyhow::Result;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

#[test]
fn test_gov_mph() -> Result<()> {
    // Read the offsets
    let m = sux::mph::gov::GOVMPH::load("tests/data/test.cmph")?;
    let reader = BufReader::new(File::open("tests/data/mph.txt")?);
    let mut s = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        let p = m.get_byte_array(line.as_bytes());
        assert!(p < m.size());
        assert!(s.insert(p));
    }
    assert_eq!(s.len(), m.size() as usize);
    Ok(())
}

#[test]
fn test_gov3_sf() -> Result<()> {
    // Read the offsets
    let m = sux::sf::gov3::GOV3::load("tests/data/test.csf")?;
    let reader = BufReader::new(File::open("tests/data/mph.txt")?);
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let p = m.get_byte_array(line.as_bytes());
        assert_eq!(p, idx as _);
    }
    Ok(())
}
