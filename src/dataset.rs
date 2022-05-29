use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::str::FromStr;
use std::io::{BufRead, BufReader};
use std::path::{Iter, Path, PathBuf};

enum Vocabulary{
    Vocab(String),
    Embedding(String),
    Empty
}

#[allow(non_snake_case)]
pub struct DataConfig{
    UNK: String,
    PAD: String,
    vocab_type: Vocabulary,
    max_length: usize
}

impl DataConfig{

    #[allow(non_snake_case)]
    pub fn new(UNK_TOKEN: String, PAD_TOKEN: String, max_length: usize) -> Self{
        Self{
            UNK: UNK_TOKEN,
            PAD: PAD_TOKEN,
            vocab_type: Vocabulary::Empty,
            max_length
        }
    }
}

impl Default for DataConfig{
    fn default() -> Self {
        Self{
            UNK: String::from("<UNK>"),
            PAD: String::from("<PAD>"),
            vocab_type: Vocabulary::Empty,
            max_length: 32
        }
    }
}
// classifier
pub struct ClassifierDataset<'a>{
    path: & 'a Path,
    train_file: PathBuf,
    dev_file: PathBuf,
    test_file: PathBuf,
    class_file: PathBuf,
    vocab_file: Option<PathBuf>,
    config: DataConfig
}

impl<'a> ClassifierDataset<'a>{
    pub fn new(path: & 'a Path) -> Self {
        Self{
            path,
            train_file: path.join("train.txt"),
            dev_file: path.join("dev.txt"),
            test_file: path.join("dev.txt"),
            class_file: path.join("class.txt"),
            vocab_file: None,
            config: DataConfig::default(),
        }
    }
    pub fn with_config(path:&'a Path, config: DataConfig) -> Self{
        Self{
            path,
            train_file: path.join("train.txt"),
            dev_file: path.join("dev.txt"),
            test_file: path.join("dev.txt"),
            class_file: path.join("class.txt"),
            vocab_file: match &config.vocab_type {
                Vocabulary::Vocab(file) => {
                    Some(path.join(file))
                }
                Vocabulary::Embedding(file) => {
                    Some(path.join(file))
                }
                Vocabulary::Empty => None
            },
            config
        }
    }
    fn train_iter(&self) -> ClassifierIter{
        let file = File::open(self.train_file.as_path()).expect("open train file failed");
        ClassifierIter::new(BufReader::new(file))
    }
    fn dev_iter(&self) -> ClassifierIter{
        let file = File::open(self.dev_file.as_path()).expect("open dev file failed");
        ClassifierIter::new(BufReader::new(file))
    }
    fn test_iter(&self) -> ClassifierIter{
        let file = File::open(self.test_file.as_path()).expect("open dev file failed");
        ClassifierIter::new(BufReader::new(file))
    }
    fn read_classes(&self) ->Vec<String>{
        let file = File::open(self.class_file.as_path()).expect("open class file failed");
        BufReader::new(file)
            .lines()
            .filter_map(|line|line.ok())
            .collect()
    }
    pub fn build_dataset<const N: usize>(&self) ->(
        Vec<ClassifierRecord<N>>,
        Vec<ClassifierRecord<N>>,
        Vec<ClassifierRecord<N>>,
        BTreeMap<String, usize>
    ){
        let class2id = self.read_classes()
            .into_iter()
            .enumerate()
            .map(|(i, class)|(class, i))
            .collect::<HashMap<_, _>>();
        let mut vocab_corpus = Vec::new();
        let mut train_samples = self.train_iter()
            .into_iter()
            .map(|s|{
                vocab_corpus.push(s.text.clone());
                s
            })
            .collect::<Vec<_>>();
        let vocab = vocab_corpus
            .iter()
            .map(|text|text.chars())
            .flatten()
            .enumerate()
            .map(|(i, ch)|(ch.to_string(), i+1))
            .collect::<BTreeMap<_, _>>();
        let train_records = Self::samples_to_records::<N>(train_samples, &vocab, &class2id);
        let mut dev_samples = self.dev_iter().into_iter().collect::<Vec<_>>();
        let dev_records = Self::samples_to_records::<N>(dev_samples, &vocab, &class2id);
        let mut test_samples = self.test_iter().into_iter().collect::<Vec<_>>();
        let test_records = Self::samples_to_records::<N>(test_samples, &vocab, &class2id);
        (train_records, dev_records, test_records, vocab)
    }
    fn samples_to_records<const N: usize>(
        samples: Vec<ClassifierSample>,
        vocab: &BTreeMap<String, usize>,
        class2id: & HashMap<String, usize>) -> Vec<ClassifierRecord<N>>{
        samples
            .into_iter()
            .map(|s|(
                s.text.chars()
                    .into_iter()
                    .map(|c|*vocab.get(&c.to_string()).unwrap_or(&0))
                    .collect(),
                s.label
                    .parse::<usize>()
                    .map_err(|e|{
                        eprintln!("num {:?} parse error {}", s.label, e);
                        e
                    })
                    // .unwrap_or(
                    //     *class2id.get(&s.label)
                    //         .expect(&format!("invalid label {}", &s.label))
                    // )
                    .unwrap_or_else(|e|{
                        *class2id.get(&s.label)
                            .expect(&format!("invalid label {}, error {}", &s.label, e))
                    }
                    )
            )
            )
            .map(|(word_ids, label_id)|ClassifierRecord::new(word_ids, label_id))
            .collect()
    }
}

struct ClassifierIter{
    reader: BufReader<File>
}

impl Iterator for ClassifierIter{
    type Item = ClassifierSample;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line){
            Ok(len) => {
                if len > 0 {
                    Some(ClassifierSample::from(line.trim_end()))
                }else {
                    None
                }
            },
            Err(e) => {
                eprintln!("{}", e);
                None
            }
        }
    }
}

impl ClassifierIter{
    fn new(reader: BufReader<File>) ->Self{
        Self{
            reader
        }
    }
}
#[derive(Debug)]
pub struct ClassifierRecord<const N: usize>{
    word_ids: [usize; N],
    label_id: usize
}

impl<const N: usize> ClassifierRecord<N> {
    fn new(mut word_ids: Vec<usize>, label_id: usize) ->Self{
        let len = word_ids.len();
        if len < N{
            word_ids.extend(vec![0; N - len])
        }else if len > N{
            word_ids.truncate(N);
        }
        Self{
            word_ids: word_ids.try_into().unwrap(),
            label_id
        }
    }
}

#[derive(Debug)]
struct  ClassifierSample{
    text: String,
    label: String,
}
impl ClassifierSample{
    fn new(text: String, label: String) ->Self{
        Self{
            text,
            label
        }
    }
}

impl From<String> for ClassifierSample{
    fn from(content: String) -> Self {
        content.split_once('\t')
            .map(|(front, back)|{
                ClassifierSample::new(front.to_string(), back.to_string())
            }).expect("invalid classifier line")
    }
}

impl From<& str> for ClassifierSample {
    fn from(content: &str) -> Self {
        content.split_once('\t')
            .map(|(front, back)|{
                ClassifierSample::new(front.to_string(), back.to_string())
            }).expect("invalid classifier line")
    }
}
// tagging
struct TaggingDataset<'a>{
    path: & 'a Path,
    config: DataConfig
}

impl<'a> TaggingDataset<'a> {
    pub fn new(path: & 'a Path) -> Self {
        Self{
            path,
            config: DataConfig::default(),
        }
    }
    pub fn with_config(path:&'a Path, config: DataConfig) -> Self{
        Self{
            path,
            config
        }
    }
}
struct TaggingIter{
    reader: BufReader<File>
}
impl TaggingIter{
    fn new(reader: BufReader<File>)->Self{
        Self{
            reader
        }
    }
}

impl Iterator for TaggingIter{
    type Item = TaggingSample;

    fn next(&mut self) -> Option<Self::Item> {
        let mut lines = Vec::new();
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line){
                Ok(len) => {
                    if len == 0{
                        break
                    }
                    lines.push(line);
                }
                Err(e) => {
                    eprintln!("{}", e);
                    return None
                }
            }
        }
        Some(TaggingSample::from(lines))
    }
}
struct TaggingSample{
    items: Vec<(String, String)>
}

impl TaggingSample{
    fn new(items: Vec<(String, String)>) ->Self{
        Self{
            items
        }
    }
}

impl From<Vec<String>> for TaggingSample {
    fn from(contents: Vec<String>) -> Self {
        let items = contents.into_iter()
            .map(|content|{
                content.trim_end().split_once('\t')
                    .map(|(front, back)|(front.to_string(), back.to_string()))
                    .expect("invalid tagging line")
            }).collect::<Vec<_>>();
        TaggingSample::new(items)
    }
}

// similarity
struct SimilarityDataset<'a>{
    path: & 'a Path,
    config: DataConfig
}

impl <'a> SimilarityDataset<'a> {
    pub fn new(path: & 'a Path) -> Self {
        Self{
            path,
            config: DataConfig::default(),
        }
    }
    pub fn with_config(path:&'a Path, config: DataConfig) -> Self{
        Self{
            path,
            config
        }
    }
}
struct SimilarityIter{
    reader: BufReader<File>
}

impl SimilarityIter {
    pub fn new(reader: BufReader<File>) ->Self{
        Self{
            reader
        }
    }
}

impl Iterator for SimilarityIter{
    type Item = SimilaritySample;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line){
            Ok(len) => {
                if len > 0{
                    return Some(SimilaritySample::from(line))
                }
            }
            Err(e) => {
                eprintln!("{}", e);
                return None
            }
        }
        None
    }
}
struct SimilaritySample{
    text_a: String,
    text_b: String,
    similar: bool
}

impl SimilaritySample{
    fn new(text_a: String, text_b: String, similar: bool) ->Self{
        Self{
            text_a,
            text_b,
            similar
        }
    }
}

impl From<String> for SimilaritySample{
    fn from(content: String) -> Self {
        let mut sp = content.splitn(3, |c| c == '\t');
        let text_a = sp.next().expect("invalid similarity sample");
        let text_b = sp.next().expect("invalid similarity sample");
        let similar = sp.next()
            .expect("invalid similarity sample")
            .parse::<u8>()
            .expect("invalid similarity tag") != 0;
        SimilaritySample::new(text_a.to_string(), text_b.to_string(), similar)
    }
}