import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from config import batch_size, n_dims, n_points, nn_hidden_dim, nn_input_dims, nn_output_dim


def extract_attention_weights(model, xs, ys, layer_idx=None):
    """
    Extract attention weights from the transformer model.
    
    Args:
        model: The transformer model
        xs: Input sequences (B, N, D)
        ys: Target sequences (B, N, output_dim)
        layer_idx: Specific layer to extract from (None for all layers)
    
    Returns:
        Dictionary with attention weights for each layer
    """
    model.eval()
    attention_weights = {}
    
    with torch.no_grad():
        zs = model._combine(xs, ys)
        embeds = model._read_in(zs)
        
        # Forward through backbone and capture attention
        outputs = model._backbone(
            inputs_embeds=embeds,
            output_attentions=True
        )
        
        # Extract attention from all layers
        attentions = outputs.attentions  # List of (B, num_heads, seq_len, seq_len)
        
        if layer_idx is not None:
            attention_weights[f'layer_{layer_idx}'] = attentions[layer_idx].cpu()
        else:
            for i, attn in enumerate(attentions):
                attention_weights[f'layer_{i}'] = attn.cpu()
    
    model.train()
    return attention_weights


def visualize_attention_patterns(attention_weights, layer_idx=0, head_idx=0, batch_idx=0, 
                                 save_path=None, title="Attention Pattern"):
    """
    Visualize attention patterns as a heatmap.
    
    Args:
        attention_weights: Dictionary of attention weights from extract_attention_weights
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize
        batch_idx: Batch index to visualize
        save_path: Path to save the figure (optional)
        title: Title for the plot
    """
    layer_key = f'layer_{layer_idx}'
    if layer_key not in attention_weights:
        raise ValueError(f"Layer {layer_idx} not found in attention_weights")
    
    attn = attention_weights[layer_key]  # (B, num_heads, seq_len, seq_len)
    attn_matrix = attn[batch_idx, head_idx].numpy()  # (seq_len, seq_len)
    
    seq_len = attn_matrix.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Add grid lines to separate x and y tokens
    for i in range(0, seq_len, 2):
        plt.axhline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Label positions
    labels = []
    for i in range(seq_len):
        if i % 2 == 0:
            labels.append(f'x{i//2}')
        else:
            labels.append(f'y{i//2}')
    plt.xticks(range(seq_len), labels, rotation=45)
    plt.yticks(range(seq_len), labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_attention_to_query(attention_weights, layer_idx=-1):
    """
    Analyze how much attention is paid to each position when predicting the query.
    
    Args:
        attention_weights: Dictionary of attention weights
        layer_idx: Layer to analyze (-1 for last layer)
    
    Returns:
        Dictionary with attention statistics
    """
    if layer_idx == -1:
        # Get last layer
        layer_keys = sorted([k for k in attention_weights.keys() if k.startswith('layer_')])
        layer_key = layer_keys[-1]
    else:
        layer_key = f'layer_{layer_idx}'
    
    attn = attention_weights[layer_key]  # (B, num_heads, seq_len, seq_len)
    B, num_heads, seq_len, _ = attn.shape
    
    # Query position is the last position (index seq_len - 1)
    query_pos = seq_len - 1
    
    # Average attention across heads and batches
    query_attention = attn[:, :, query_pos, :].mean(dim=(0, 1))  # (seq_len,)
    
    # Separate attention to x vs y positions
    x_attention = query_attention[::2].sum().item()  # Sum over all x positions
    y_attention = query_attention[1::2].sum().item()  # Sum over all y positions
    
    # Attention to most recent examples
    recent_k = min(5, seq_len // 2)
    recent_attention = query_attention[-recent_k*2:].sum().item()
    
    return {
        'total_attention': query_attention.sum().item(),
        'x_attention': x_attention,
        'y_attention': y_attention,
        'recent_attention': recent_attention,
        'attention_distribution': query_attention.numpy(),
    }


def compare_attention_across_layers(attention_weights, head_idx=0, batch_idx=0, save_path=None):
    """
    Compare attention patterns across all layers.
    
    Args:
        attention_weights: Dictionary of attention weights
        head_idx: Head index to visualize
        batch_idx: Batch index to visualize
        save_path: Path to save the figure
    """
    layer_keys = sorted([k for k in attention_weights.keys() if k.startswith('layer_')])
    n_layers = len(layer_keys)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    
    for idx, layer_key in enumerate(layer_keys):
        attn = attention_weights[layer_key]
        attn_matrix = attn[batch_idx, head_idx].numpy()
        
        im = axes[idx].imshow(attn_matrix, cmap='Blues', aspect='auto')
        axes[idx].set_title(f'Layer {idx}')
        axes[idx].set_xlabel('Key Position')
        axes[idx].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle(f'Attention Patterns Across Layers (Head {head_idx})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Multi-layer attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_attention_entropy(attention_weights):
    """
    Calculate attention entropy to measure how focused vs uniform the attention is.
    Higher entropy = more uniform attention, lower entropy = more focused.
    
    Args:
        attention_weights: Dictionary of attention weights
    
    Returns:
        Dictionary mapping layer to entropy statistics
    """
    entropies = {}
    
    for layer_key, attn in attention_weights.items():
        # attn: (B, num_heads, seq_len, seq_len)
        B, num_heads, seq_len, _ = attn.shape
        
        # Calculate entropy for each head and position
        # Entropy = -sum(p * log(p)) where p is attention distribution
        eps = 1e-10  # Small epsilon to avoid log(0)
        attn_normalized = attn + eps
        attn_normalized = attn_normalized / attn_normalized.sum(dim=-1, keepdim=True)
        
        entropy = -(attn_normalized * torch.log(attn_normalized + eps)).sum(dim=-1)
        # entropy: (B, num_heads, seq_len)
        
        # Average across batches, heads, and positions
        mean_entropy = entropy.mean().item()
        std_entropy = entropy.std().item()
        
        # Entropy for query position specifically
        query_entropy = entropy[:, :, -1].mean().item()
        
        entropies[layer_key] = {
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'query_entropy': query_entropy,
        }
    
    return entropies


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("Generating test data...")
    xs, ys = generate_linear(n_points, batch_size=1, n_dims=n_dims)
    
    print("Extracting attention weights...")
    attention_weights = extract_attention_weights(model, xs, ys)
    
    print("Visualizing attention patterns...")
    visualize_attention_patterns(
        attention_weights, 
        layer_idx=0, 
        head_idx=0, 
        batch_idx=0,
        title="First Layer Attention"
    )
    
    print("Analyzing attention to query...")
    query_analysis = analyze_attention_to_query(attention_weights, layer_idx=-1)
    print(f"X attention: {query_analysis['x_attention']:.4f}")
    print(f"Y attention: {query_analysis['y_attention']:.4f}")
    print(f"Recent attention: {query_analysis['recent_attention']:.4f}")
    
    print("Calculating attention entropy...")
    entropies = analyze_attention_entropy(attention_weights)
    for layer_key, stats in entropies.items():
        print(f"{layer_key}: Mean Entropy = {stats['mean_entropy']:.4f}, Query Entropy = {stats['query_entropy']:.4f}")

