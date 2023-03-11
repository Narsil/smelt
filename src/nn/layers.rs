use crate::ops::{add, matmul, matmul_t, mul, normalize, select};
use crate::tensor::{OwnedTensor, ViewTensor};
use crate::SmeltError;

/// Linear layer, performs matmul(X, A.t) + B (like PyTorch)
#[derive(Clone)]
pub struct Linear<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
}

impl<'a> Linear<'a> {
    /// Creates a new Linear layer, no check of the incoming tensors.
    pub fn new(weight: ViewTensor<'a>, bias: ViewTensor<'a>) -> Self {
        Self { weight, bias }
    }

    /// See the weight tensor
    pub fn weight(&'a self) -> &'a ViewTensor<'a> {
        &self.weight
    }

    /// Puts the results of matmul_t(tensor, self.weight) + self.bias within out
    pub fn forward(&self, tensor: &OwnedTensor, out: &mut OwnedTensor) -> Result<(), SmeltError> {
        matmul_t(tensor, &self.weight, out)?;
        add(&self.bias, out)
    }
}

/// Linear layer, performs matmul(X, A.t) + B (**not** like PyTorch)
#[derive(Clone)]
pub struct LinearT<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
}

impl<'a> LinearT<'a> {
    /// Creates a new LinearT layer, no check of the incoming tensors.
    pub fn new(weight: ViewTensor<'a>, bias: ViewTensor<'a>) -> Self {
        Self { weight, bias }
    }

    /// Puts the results of matmul(tensor, self.weight) + self.bias within out
    pub fn forward(
        &self,
        tensor: &mut OwnedTensor,
        out: &mut OwnedTensor,
    ) -> Result<(), SmeltError> {
        matmul(tensor, &self.weight, out)?;
        add(&self.bias, out)
    }
}

/// Embedding layer
#[derive(Clone)]
pub struct Embedding<'a> {
    weight: ViewTensor<'a>,
}

impl<'a> Embedding<'a> {
    /// Creates a new Embedding layer, without checks
    pub fn new(weight: ViewTensor<'a>) -> Self {
        Self { weight }
    }

    /// See the weight tensor
    pub fn weight(&'a self) -> &'a ViewTensor<'a> {
        &self.weight
    }

    /// Puts the rows from self.weight following `ids` into `out` tensor.
    pub fn forward(&self, ids: &[usize], out: &mut OwnedTensor) -> Result<(), SmeltError> {
        select(ids, &self.weight, out)
    }
}

/// Layer Norm layer
#[derive(Clone)]
pub struct LayerNorm<'a> {
    weight: ViewTensor<'a>,
    bias: ViewTensor<'a>,
    epsilon: f32,
}

impl<'a> LayerNorm<'a> {
    /// Layer Norm layer without checks
    pub fn new(weight: ViewTensor<'a>, bias: ViewTensor<'a>, epsilon: f32) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    /// Applies (tensor - mean(tensor)) / (var(tensor) + epsilon) over the last dimension of `tensor`.
    pub fn forward(
        &self,
        tensor: &mut OwnedTensor,
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<(), SmeltError> {
        normalize(tensor, mean, var, self.epsilon)?;
        mul(&self.weight, tensor)?;
        add(&self.bias, tensor)
    }
}
