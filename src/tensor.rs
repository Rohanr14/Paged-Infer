//! Raw checkpoint tensors: memory-mapped bytes plus what is needed to decode
//! them.
//!
//! A tensor carries its element type. The loader used to keep only bytes and
//! shape and read everything as bf16, so an f16 or f32 checkpoint loaded
//! "successfully" and ran on garbage. Every read goes through [`DType`] now,
//! and anything the engine cannot decode is refused before it is ever used.

/// Element type of a checkpoint tensor, restricted to what the engine decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
}

impl DType {
    /// Map a safetensors dtype, or `None` for one the engine does not decode.
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Option<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Some(DType::BF16),
            safetensors::Dtype::F16 => Some(DType::F16),
            safetensors::Dtype::F32 => Some(DType::F32),
            _ => None,
        }
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            DType::BF16 | DType::F16 => 2,
            DType::F32 => 4,
        }
    }

    /// Decode `out.len()` consecutive elements from `bytes`.
    pub fn decode_into(self, bytes: &[u8], out: &mut [f32]) {
        assert_eq!(
            bytes.len(),
            out.len() * self.size(),
            "{self:?}: byte length does not match element count"
        );
        match self {
            DType::BF16 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
            DType::F16 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
            DType::F32 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
    }
}

/// A bare-metal tensor over memory-mapped weight bytes.
#[derive(Debug, Clone)]
pub struct Tensor<'data> {
    data: &'data [u8],
    shape: Vec<usize>,
    dtype: DType,
}

impl<'data> Tensor<'data> {
    /// # Panics
    ///
    /// If `data` is not exactly `shape.product() * dtype.size()` bytes: a
    /// mismatch here means a later read would run off the end or read the
    /// wrong element, and neither should be discovered at inference time.
    pub fn new(data: &'data [u8], shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel * dtype.size(),
            "tensor of shape {shape:?} ({dtype:?}) does not match {} bytes",
            data.len()
        );
        Self { data, shape, dtype }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn raw_bytes(&self) -> &'data [u8] {
        self.data
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Widen every element to f32.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.numel()];
        self.dtype.decode_into(self.data, &mut out);
        out
    }

    /// Decode row `row` of a two-dimensional tensor into `out`.
    ///
    /// # Panics
    ///
    /// If `row` is out of range. For the embedding table that is a token id
    /// outside the vocabulary — which the engine refuses at submission, so
    /// reaching this is a bug, and reading a neighbouring row would hide it.
    pub fn row_into(&self, row: usize, out: &mut [f32]) {
        assert_eq!(self.shape.len(), 2, "row_into needs a 2-D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        assert!(row < rows, "row {row} out of range for {rows} rows");
        assert_eq!(out.len(), cols);
        let width = cols * self.dtype.size();
        self.dtype
            .decode_into(&self.data[row * width..(row + 1) * width], out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_each_dtype_decodes_its_own_encoding() {
        let values = [1.5f32, -0.25, 3.0, 0.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let f16: Vec<u8> = values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let f32b: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        for (dtype, bytes) in [(DType::BF16, bf16), (DType::F16, f16), (DType::F32, f32b)] {
            let t = Tensor::new(&bytes, vec![2, 2], dtype);
            assert_eq!(t.to_f32_vec(), values, "{dtype:?}");
            let mut row = [0.0; 2];
            t.row_into(1, &mut row);
            assert_eq!(row, [3.0, 0.0], "{dtype:?} row 1");
        }
    }

    #[test]
    fn test_reading_f16_bytes_as_bf16_is_not_the_same_number() {
        // The bug this module exists to prevent: the two half formats share a
        // width and nothing else.
        let one = half::f16::from_f32(1.0).to_le_bytes();
        assert_ne!(half::bf16::from_le_bytes(one).to_f32(), 1.0);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn test_length_mismatch_is_caught_at_construction() {
        let bytes = [0u8; 6];
        let _ = Tensor::new(&bytes, vec![2, 2], DType::BF16);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_row_past_the_table_is_a_panic_not_a_neighbour() {
        let bytes = [0u8; 8];
        let t = Tensor::new(&bytes, vec![2, 2], DType::BF16);
        let mut row = [0.0; 2];
        t.row_into(2, &mut row);
    }
}
