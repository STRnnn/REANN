import torch
import sys

def _get_io_dims(module):
    """A helper function to determine the I/O dimensions of a module."""
    module_type = module.__class__.__name__
    input_dim, output_dim = None, None

    if module_type == 'Linear':
        input_dim, output_dim = module.in_features, module.out_features
    elif module_type == 'GetDensity':
        if hasattr(module, 'hyper'):
            output_dim = module.hyper.shape[-1]
    elif module_type in ['Relu_like', 'Tanh_like']:
        if hasattr(module, 'alpha'):
            input_dim = output_dim = module.alpha.shape[1]
    elif module_type == 'LayerNorm':
        if hasattr(module, 'normalized_shape'):
            input_dim = output_dim = module.normalized_shape[0]
    
    return input_dim, output_dim

def print_model_summary(model, file=None):
    """
    Prints a concise summary of a PyTorch model's layers, dimensions,
    and parameter counts, filtering out container modules.
    """
    if file is None:
        file = sys.stdout

    header = f"{'name':<55} {'type':<20} {'input':<10} {'output':<10} {'num_params':<12}"
    file.write(header + "\n")
    
    for name, m in model.named_modules():
        if not name: continue

        # Skip containers: modules whose direct children have parameters.
        if any(sum(p.numel() for p in c.parameters()) > 0 for c in m.children()):
            continue
        
        params = sum(p.numel() for p in m.parameters())
        if params == 0: continue
        
        module_type = m.__class__.__name__
        input_dim, output_dim = _get_io_dims(m)
        
        input_str = str(input_dim) if input_dim is not None else ""
        output_str = str(output_dim) if output_dim is not None else ""
        
        line = f"{name:<55} {module_type:<20} {input_str:<10} {output_str:<10} {params:<12,}"
        file.write(line + "\n")

    total_para = sum(p.numel() for p in model.parameters())
    file.write(f"total_para:{total_para}\n")
