# Qwen2 Architecture Implementation

This is an implementation of the Qwen2 architecture in C, based on the llama2.c codebase. The Qwen2 architecture includes several improvements over the original Llama2 architecture, making it more efficient and capable of handling longer contexts.

## Key Features

1. **Group Query Attention (GQA)**: Improved implementation of GQA for faster inference and reduced memory usage.
2. **Sliding Window Attention**: Support for sliding window attention to efficiently handle long contexts up to 128K tokens.
3. **Enhanced RoPE (Rotary Positional Embeddings)**: Configurable RoPE dimension for better position encoding in long sequences.
4. **Qwen2 Chat Format**: Support for the Qwen2 chat format with system and user messages.
5. **SwiGLU Activation**: Optimized implementation of the SwiGLU activation function used in the feed-forward network.

## Usage

Compile the code using the provided Makefile:

```bash
make run
```

This will create the `runqwen2` executable.

### Running the Model

```bash
./runqwen2 <checkpoint> [options]
```

### Options

- `-t <float>`: Temperature in [0,inf], default 1.0
- `-p <float>`: P value in top-p (nucleus) sampling in [0,1], default 0.9
- `-s <int>`: Random seed, default time(NULL)
- `-n <int>`: Number of steps to run for, default 256. 0 = max_seq_len
- `-i <string>`: Input prompt
- `-z <string>`: Optional path to custom tokenizer
- `-m <string>`: Mode: generate|chat, default: generate
- `-y <string>`: (Optional) system prompt in chat mode
- `-w <int>`: Sliding window size for attention (0 = disabled), default: 0
- `-r <int>`: Rotary positional embedding dimension (0 = use head_size), default: 0

### Example

```bash
./runqwen2 model.bin -n 256 -i "Once upon a time" -w 4096 -r 128
```

This runs the model with a sliding window of 4096 tokens and a RoPE dimension of 128.

## Chat Mode

In chat mode, the implementation uses the Qwen2 chat format:

```
<|im_start|>system
{system_message}
<|im_end|>

<|im_start|>user
{user_message}
<|im_end|>

<|im_start|>assistant
{assistant_message}
<|im_end|>
```

Example usage:

```bash
./runqwen2 model.bin -m chat -y "You are a helpful assistant." -i "Hello, how are you?"
```

## Technical Improvements

### 1. Group Query Attention (GQA)

The implementation optimizes the attention mechanism by using grouped queries, where multiple query heads share the same key and value heads. This reduces the computational cost and memory requirements during inference.

### 2. Sliding Window Attention

For handling long contexts efficiently, the implementation includes a sliding window attention mechanism that limits the attention span to a configurable window size. This significantly reduces the quadratic complexity of self-attention for long sequences.

### 3. Enhanced RoPE

The rotary positional embeddings have been enhanced with a configurable dimension parameter, allowing for better extrapolation to longer sequences than those seen during training.

### 4. Optimized Matrix Multiplication

The matrix multiplication operations have been optimized for the quantized model, with special consideration for the Group Query Attention pattern.

## Compatibility

This implementation is designed to be compatible with Qwen2 model weights. The checkpoint format is the same as the original llama2.c implementation, but with additional configuration parameters for the Qwen2-specific features.