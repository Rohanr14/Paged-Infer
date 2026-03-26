use std::sync::Arc;

/// A bare-metal tensor implementation to avoid heavy dependencies.
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Arc<[T]>, // Arc allows cheap cloning of memory-mapped views
    shape: Vec<usize>,
}

impl<T: Copy> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length does not match shape dimensions"
        );
        
        Self {
            data: data.into(),
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
}