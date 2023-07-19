use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use std::hint::black_box;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks construction of RCA and print stats", long_about = None)]
struct Args {
    /// The file to read, every line will be inserted in the RCA.
    file_path: String,

    #[arg(short, long, default_value = "4")]
    /// Fully write every string with index multiple of k.
    k: usize,
}

pub fn main() {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let args = Args::parse();

    let mut rca = <RearCodedArray<usize>>::new(args.k);
    let mut pl = ProgressLogger::default().display_memory();
    pl.item_name = "line";

    let lines = std::io::BufReader::new(std::fs::File::open(&args.file_path).unwrap())
        .lines()
        .map(|line| line.unwrap());

    pl.start("Inserting...");
    for line in lines {
        rca.push(line);
        pl.light_update();
    }

    pl.done();

    rca.print_stats();
}
