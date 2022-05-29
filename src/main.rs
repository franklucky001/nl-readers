mod dataset;
use std::path::Path;
use dataset::{DataConfig, ClassifierDataset};

fn build_classifier(){
    let path = Path::new("data/classifier/THUCNews");
    let dataset = ClassifierDataset::new(&path);
    let (train, dev, test, vocab) = dataset.build_dataset::<32usize>();
    println!("train {} samples", train.len());
    println!("dev {} samples", dev.len());
    println!("test {} samples", test.len());
    println!("vocab size: {}", vocab.len());
}

fn main() {
    build_classifier();
    println!("Hello, world!");
}
