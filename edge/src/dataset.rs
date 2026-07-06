pub trait DatasetLoader: Send {
    fn num_samples(&self) -> usize;
    fn feature_dim(&self) -> usize;
    fn sample(&self, index: usize) -> Option<DatasetSample>;
}

#[derive(Debug, Clone)]
pub struct DatasetSample {
    pub features: Vec<f32>,
    pub label: u32,
}

pub struct CsvDataset {
    samples: Vec<DatasetSample>,
    feature_dim: usize,
}

impl CsvDataset {
    pub fn from_path(path: &str) -> Result<Self, String> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| format!("failed to open CSV: {e}"))?;

        let mut samples = Vec::new();
        let mut feature_dim = 0;

        for result in rdr.records() {
            let record = result.map_err(|e| format!("CSV parse error: {e}"))?;
            let num_fields = record.len();
            if num_fields < 2 {
                return Err("CSV row has fewer than 2 columns".into());
            }
            let mut fields: Vec<&str> = record.iter().collect();
            let label_field = fields.pop().ok_or("no label field")?;
            let features: Vec<f32> = fields
                .iter()
                .map(|f| {
                    f.parse::<f32>()
                        .map_err(|e| format!("feature parse error: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let label: u32 = label_field
                .parse()
                .map_err(|e| format!("label parse error: {e}"))?;
            if feature_dim == 0 {
                feature_dim = features.len();
            }
            samples.push(DatasetSample { features, label });
        }

        if samples.is_empty() {
            return Err("CSV dataset is empty".into());
        }

        Ok(Self {
            samples,
            feature_dim,
        })
    }
}

impl DatasetLoader for CsvDataset {
    fn num_samples(&self) -> usize {
        self.samples.len()
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn sample(&self, index: usize) -> Option<DatasetSample> {
        self.samples.get(index).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_csv_loader() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_dataset.csv");
        let content = "f1,f2,f3,label\n0.1,0.2,0.3,0\n0.4,0.5,0.6,1\n";
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();

        let dataset = CsvDataset::from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(dataset.num_samples(), 2);
        assert_eq!(dataset.feature_dim(), 3);

        let s0 = dataset.sample(0).unwrap();
        assert_eq!(s0.label, 0);
        assert!((s0.features[0] - 0.1).abs() < 1e-6);

        let s1 = dataset.sample(1).unwrap();
        assert_eq!(s1.label, 1);

        std::fs::remove_file(&path).ok();
    }
}
