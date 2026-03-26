use std::sync::Arc;

/// A bare-metal tensor implementation to hold memory-mapped weights.
#[derive(Debug, Clone)]
pub struct Tensor<'data> {
    data: &'data [u8], // Raw memory-mapped bytes
    shape: Vec<usize>,
}

impl<'data> Tensor<'data> {
    pub fn new(data: &'data [u8], shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.data
    }
}