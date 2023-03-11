use crate::nn::layers::{Embedding, LayerNorm, Linear};
use crate::ops::{add, apply, gelu, matmul, matmul_t, select, softmax};
use crate::tensor::{OwnedTensor, Tensor, TensorMut};
use crate::SmeltError;

fn split_heads<T: Tensor>(q: &T, num_heads: usize) -> Result<OwnedTensor, SmeltError> {
    let sequence_length = q.shape()[0];
    let hidden_dim = q.shape()[1];
    assert_eq!(hidden_dim % num_heads, 0);
    let head_dim = hidden_dim / num_heads;
    let mut query_data = vec![0.0; num_heads * sequence_length * head_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let index = j * hidden_dim + i * head_dim + k;
                let out_index = i * sequence_length * head_dim + j * head_dim + k;
                let value = q.data()[index];
                query_data[out_index] = value;
            });
        });
    });
    Ok(OwnedTensor::new(
        query_data,
        vec![num_heads, sequence_length, head_dim],
    )?)
}

fn attention<T: Tensor, TM: TensorMut>(
    query: &T,
    key: &T,
    value: &T,
    qk: &mut TM,
    max: &mut [f32],
    out: &mut OwnedTensor,
) -> Result<(), SmeltError> {
    let sequence_length = query.shape()[0];
    let hidden_dim = query.shape()[1];
    let num_heads = qk.shape()[0];
    assert_eq!(hidden_dim % num_heads, 0);

    assert_eq!(
        qk.shape(),
        vec![num_heads, sequence_length, sequence_length]
    );

    let query = split_heads(query, num_heads)?;
    let key = split_heads(key, num_heads)?;
    let value = split_heads(value, num_heads)?;

    matmul_t(&query, &key, qk)?;
    let head_dim = hidden_dim / num_heads;
    let scale = (head_dim as f32).sqrt();
    qk.data_mut().iter_mut().for_each(|v| *v /= scale);

    softmax(qk, max)?;
    matmul(qk, &value, out)?;

    let mut new_out = vec![0.0; sequence_length * hidden_dim];
    (0..num_heads).for_each(|i| {
        (0..sequence_length).for_each(|j| {
            (0..head_dim).for_each(|k| {
                let in_index = i * sequence_length * head_dim + j * head_dim + k;
                let out_index = j * hidden_dim + i * head_dim + k;
                new_out[out_index] = out.data()[in_index];
            });
        });
    });
    *out = OwnedTensor::new(new_out, vec![sequence_length, hidden_dim])?;
    Ok(())
}

/// 2 Layer Mlp with larger inner dimension
#[derive(Clone)]
pub struct Mlp<'a> {
    /// Hidden -> 3x Hidden
    intermediate: Linear<'a>,
    /// 3x Hidden -> Hidden
    output: Linear<'a>,
    /// Normalization
    output_ln: LayerNorm<'a>,
}

impl<'a> Mlp<'a> {
    /// Compute the MLP
    pub fn forward(
        &self,
        tensor: &mut OwnedTensor,
        intermediate: &mut OwnedTensor,
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<(), SmeltError> {
        let input_tensor = tensor.clone();
        self.intermediate.forward(tensor, intermediate)?;
        apply(intermediate, gelu);
        self.output.forward(intermediate, tensor)?;
        add(&input_tensor, tensor)?;
        self.output_ln.forward(tensor, mean, var)
    }
}

/// The attention of the bert model
#[derive(Clone)]
pub struct BertAttention<'a> {
    query: Linear<'a>,
    key: Linear<'a>,
    value: Linear<'a>,
    output: Linear<'a>,
    output_ln: LayerNorm<'a>,
    num_heads: usize,
}

impl<'a> BertAttention<'a> {
    /// Computes the attention of the Bert model
    pub fn forward(
        &self,
        hidden_states: &mut OwnedTensor,
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<(), SmeltError> {
        // println!("---");
        //debug!("Attention", hidden_states);
        assert_eq!(hidden_states.shape().len(), 2);
        let input_tensor = hidden_states.clone();
        let sequence_length = hidden_states.shape()[0];
        let hidden_dim = hidden_states.shape()[1];
        let num_heads = self.num_heads;

        let mut q = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);
        let mut k = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);
        let mut v = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);
        self.query.forward(hidden_states, &mut q)?;
        self.key.forward(hidden_states, &mut k)?;
        self.value.forward(hidden_states, &mut v)?;

        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        let mut qk = OwnedTensor::zeros(vec![num_heads, sequence_length, sequence_length]);
        let mut qv = OwnedTensor::zeros(vec![num_heads, sequence_length, head_dim]);
        let mut max = vec![0.0; (sequence_length) * num_heads];
        attention(&q, &k, &v, &mut qk, &mut max, &mut qv)?;

        let input: OwnedTensor = qv.clone();

        self.output.forward(&input, &mut qv)?;
        add(&input_tensor, &mut qv)?;
        self.output_ln.forward(&mut qv, mean, var)?;
        *hidden_states = qv;
        Ok(())
    }
}

/// The entire bert layer
#[derive(Clone)]
pub struct BertLayer<'a> {
    mlp: Mlp<'a>,
    attention: BertAttention<'a>,
}

impl<'a> BertLayer<'a> {
    /// Compute the BertLayer
    pub fn forward(
        &self,
        tensor: &mut OwnedTensor,
        intermediate: &mut OwnedTensor,
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<(), SmeltError> {
        self.attention.forward(tensor, mean, var)?;
        self.mlp.forward(tensor, intermediate, mean, var)
    }
}

/// The Bert Encoder which creates the latent space
#[derive(Clone)]
pub struct BertEncoder<'a> {
    layers: Vec<BertLayer<'a>>,
}

impl<'a> BertEncoder<'a> {
    /// BertEncoder computation
    pub fn forward(
        &self,
        tensor: &mut OwnedTensor,
        intermediate: &mut OwnedTensor,
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<(), SmeltError> {
        for layer in &self.layers {
            layer.forward(tensor, intermediate, mean, var)?;
        }
        Ok(())
    }
}

/// Bert Pooler layer, which reduces the sequence length to 1 by selecting
/// the first hidden_state in the sequence
#[derive(Clone)]
pub struct BertPooler<'a> {
    pooler: Linear<'a>,
}

impl<'a> BertPooler<'a> {
    /// BertPooler forward
    pub fn forward(&self, tensor: &mut OwnedTensor) -> Result<(), SmeltError> {
        let mut first = OwnedTensor::zeros(vec![1, tensor.shape()[1]]);
        select(&[0], tensor, &mut first)?;
        let mut out = OwnedTensor::zeros(vec![tensor.shape()[1], self.pooler.weight().shape()[1]]);
        self.pooler.forward(&first, &mut out)?;
        apply(&mut out, f32::tanh);
        *tensor = out;
        Ok(())
    }
}

/// Bert Embeddings combining, ids, types, and positions followed by a layer norm
#[derive(Clone)]
pub struct BertEmbeddings<'a> {
    wte: Embedding<'a>,
    wpe: Embedding<'a>,
    type_embeddings: Embedding<'a>,
    layer_norm: LayerNorm<'a>,
}

impl<'a> BertEmbeddings<'a> {
    /// Bert Embeddings forward
    pub fn forward(
        &self,
        input_ids: &[usize],
        type_ids: &[usize],
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<OwnedTensor, SmeltError> {
        let mut tensor = OwnedTensor::zeros(vec![input_ids.len(), self.wte.weight().shape()[1]]);
        self.wte.forward(input_ids, &mut tensor)?;

        let mut type_embeds = tensor.clone();
        self.type_embeddings.forward(type_ids, &mut type_embeds)?;
        add(&type_embeds, &mut tensor)?;

        let positions: Vec<usize> = (0..input_ids.len()).map(|i| i as usize).collect();
        let mut position_embeds = type_embeds;
        self.wpe.forward(&positions[..], &mut position_embeds)?;
        add(&position_embeds, &mut tensor)?;
        self.layer_norm.forward(&mut tensor, mean, var)?;
        Ok(tensor)
    }
}

/// The entire Bert model
#[derive(Clone)]
pub struct Bert<'a> {
    embeddings: BertEmbeddings<'a>,
    encoder: BertEncoder<'a>,
    pooler: BertPooler<'a>,
    classifier: Linear<'a>,
}

impl<'a> Bert<'a> {
    /// The entire Bert forward
    pub fn forward(
        &self,
        input_ids: &[usize],
        type_ids: &[usize],
        mean: &mut [f32],
        var: &mut [f32],
    ) -> Result<OwnedTensor, SmeltError> {
        let mut tensor = self.embeddings.forward(input_ids, type_ids, mean, var)?;
        let mut intermediate = OwnedTensor::zeros(vec![768, self.classifier.weight().shape()[1]]);
        self.encoder
            .forward(&mut tensor, &mut intermediate, mean, var)?;
        self.pooler.forward(&mut tensor)?;
        let mut logits = OwnedTensor::zeros(vec![1, self.classifier.weight().shape()[1]]);
        self.classifier.forward(&tensor, &mut logits)?;
        softmax(&mut logits, mean)?;
        Ok(logits)
    }
}
