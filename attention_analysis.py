import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from config import batch_size, n_dims, n_points, nn_hidden_dim, nn_input_dims, nn_output_dim


def extract_attention_weights(model, xs, ys, layer_idx=None):
    """
    Extract attention weights from the transformer model.
    Note: Model must be configured with output_attentions=True
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
        if outputs.attentions is not None:
            attentions = outputs.attentions
            
            if layer_idx is not None:
                attention_weights[f'layer_{layer_idx}'] = attentions[layer_idx].cpu()
            else:
                for i, attn in enumerate(attentions):
                    attention_weights[f'layer_{i}'] = attn.cpu()
        else:
            print("Warning: Model did not return attention weights. Ensure output_attentions=True in config.")
    
    model.train()
    return attention_weights


def visualize_attention_patterns(attention_weights, layer_idx=0, head_idx=0, batch_idx=0, 
                                 save_path=None, title="Attention Pattern"):
    """Visualize attention patterns as a heatmap."""
    layer_key = f'layer_{layer_idx}'
    if layer_key not in attention_weights:
        print(f"Layer {layer_idx} not found in attention_weights")
        return
    
    attn = attention_weights[layer_key]
    attn_matrix = attn[batch_idx, head_idx].numpy()
    
    seq_len = attn_matrix.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Add grid lines
    for i in range(0, seq_len, 2):
        plt.axhline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_attention_to_query(attention_weights, layer_idx=-1):
    """Analyze how much attention the query position receives."""
    if layer_idx == -1:
        layer_keys = sorted([k for k in attention_weights.keys() if k.startswith('layer_')])
        if not layer_keys:
            return {}
        layer_key = layer_keys[-1]
    else:
        layer_key = f'layer_{layer_idx}'
    
    if layer_key not in attention_weights:
        return {}
    
    attn = attention_weights[layer_key]
    B, num_heads, seq_len, _ = attn.shape
    
    # Query position is the last position
    query_pos = seq_len - 1
    query_attention = attn[:, :, query_pos, :].mean(dim=(0, 1))
    
    # Separate x vs y positions
    x_attention = query_attention[::2].sum().item()
    y_attention = query_attention[1::2].sum().item()
    
    return {
        'total_attention': query_attention.sum().item(),
        'x_attention': x_attention,
        'y_attention': y_attention,
        'attention_distribution': query_attention.numpy(),
    }


if __name__ == "__main__":
    print("Loading model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("Generating test data...")
    xs, ys = generate_linear(n_points, batch_size=1, n_dims=n_dims)
    
    print("Extracting attention weights...")
    attention_weights = extract_attention_weights(model, xs, ys)
    
    if attention_weights:
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
        print(f"X attention: {query_analysis.get('x_attention', 0):.4f}")
        print(f"Y attention: {query_analysis.get('y_attention', 0):.4f}")
    else:
        print("No attention weights extracted. Model needs output_attentions=True in GPT2Config.")
