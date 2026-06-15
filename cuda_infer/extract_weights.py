#!/usr/bin/env python3
"""
extract_weights.py — Extract weights from Qwen3.5-35B-A3B-GPTQ-Int4 into a flat binary.

Outputs:
  - model_weights.bin: binary blob with embedded JSON manifest header

Binary format:
  [offset]  [size]  [description]
  0x0000   header_size (uint32) - size of JSON manifest
  ...      JSON manifest (variable size, padded to 64-byte alignment)
  ...      tensor data

Usage:
    python extract_weights.py <model_dir> <output_bin>
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def read_tensor_data(filepath, header, data_start, tensor_name):
    """Read tensor data from a safetensors file."""
    meta = header[tensor_name]
    tensor_offsets = meta['data_offsets']
    byte_len = tensor_offsets[1] - tensor_offsets[0]

    with open(filepath, 'rb') as f:
        f.seek(data_start + tensor_offsets[0])
        data = f.read(byte_len)

    return data, meta['shape'], meta['dtype']


def get_tensor_short_name(name):
    """Extract the tensor name relative to its section, for manifest."""
    # Handle lm_head.weight at root level
    if name == 'lm_head.weight':
        return 'lm_head.weight'
    # Remove model.language_model. prefix
    if name.startswith('model.language_model.'):
        return name[len('model.language_model.'):]
    return name


def get_category(san_name):
    """Determine the category for a sanitized tensor name."""
    if san_name == 'embed_tokens.weight':
        return 'embed_tokens'
    elif san_name == 'lm_head.weight':
        return 'lm_head'
    elif san_name == 'norm.weight':
        return 'final_layer_norm'
    elif san_name.startswith('layers.'):
        # Extract layer number
        parts = san_name.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                return f'layer_{parts[i+1]}'
        return 'layer_unknown'
    else:
        return 'other'


def main():
    parser = argparse.ArgumentParser(description='Extract weights to binary')
    parser.add_argument('model_dir', type=str,
                        help='Path to model directory')
    parser.add_argument('output_bin', type=str,
                        help='Output binary file path')
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_bin = Path(args.output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    # Load weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']
    print(f"Total weights in index: {len(weight_map)}")

    # Group tensors by shard file
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        by_file[filename].append(name)

    # Parse all safetensors headers
    print("Parsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    # Categorize and extract tensors
    print("\nExtracting weights...")

    # Collect all language_model tensors we need to extract
    all_tensors = []  # (san_name, orig_name, filename, category)

    for orig_name, filename in weight_map.items():
        san_name = get_tensor_short_name(orig_name)
        if san_name is None:
            continue

        category = get_category(san_name)
        all_tensors.append((san_name, orig_name, filename, category))

    # Sort by category then name
    def sort_key(item):
        san_name, orig_name, filename, category = item
        if category == 'embed_tokens':
            return (0, san_name)
        elif category == 'lm_head':
            return (1, san_name)
        elif category == 'final_layer_norm':
            return (2, san_name)
        elif category.startswith('layer_'):
            layer_num = int(category.split('_')[1]) if category.split('_')[1].isdigit() else 999
            return (3, layer_num, san_name)
        else:
            return (4, san_name)

    all_tensors.sort(key=sort_key)

    # First pass: determine offsets
    print("Planning layout...")
    t0 = time.time()

    ALIGN = 64
    offset = 0
    layout = []  # (san_name, orig_name, filename, offset, size, shape, category)

    for san_name, orig_name, filename, category in all_tensors:
        filepath = model_path / filename
        header, _ = header_cache[filename]

        if orig_name not in header:
            print(f"  WARNING: {orig_name} not found in {filename}, skipping")
            continue

        meta = header[orig_name]
        tensor_offsets = meta['data_offsets']
        byte_len = tensor_offsets[1] - tensor_offsets[0]

        # Align offset
        if offset % ALIGN != 0:
            pad = ALIGN - (offset % ALIGN)
            offset += pad

        layout.append((san_name, orig_name, filename, offset, byte_len,
                       meta['shape'], category))

        offset += byte_len

    total_bytes = offset

    print(f"Total weight data: {total_bytes / 1e9:.2f} GB")

    # Build manifest with computed offsets
    manifest = {
        "embed_tokens": {},
        "lm_head": {},
        "final_layer_norm": {},
        "layers": {}
    }

    for san_name, orig_name, filename, off, size, shape, category in layout:
        if category == 'embed_tokens':
            manifest['embed_tokens'] = {"offset": off, "size": size, "shape": shape}
        elif category == 'lm_head':
            manifest['lm_head'] = {"offset": off, "size": size, "shape": shape}
        elif category == 'final_layer_norm':
            manifest['final_layer_norm'] = {"offset": off, "size": size, "shape": shape}
        elif category.startswith('layer_'):
            layer_idx = int(category.split('_')[1])
            if layer_idx not in manifest['layers']:
                manifest['layers'][layer_idx] = {}
            manifest['layers'][layer_idx][san_name] = {"offset": off, "size": size, "shape": shape}

    # Build JSON manifest and calculate header size
    manifest_json = json.dumps(manifest, indent=2)
    json_size = len(manifest_json.encode('utf-8'))

    # Header layout: 4 bytes header_size + JSON + padding to 64-byte alignment
    header_base = 4  # header_size field
    data_offset = (json_size + header_base + (ALIGN - 1)) & ~(ALIGN - 1)

    # Update manifest offsets to account for embedded header
    for san_name, orig_name, filename, off, size, shape, category in layout:
        if category == 'embed_tokens':
            manifest['embed_tokens'] = {"offset": off + data_offset, "size": size, "shape": shape}
        elif category == 'lm_head':
            manifest['lm_head'] = {"offset": off + data_offset, "size": size, "shape": shape}
        elif category == 'final_layer_norm':
            manifest['final_layer_norm'] = {"offset": off + data_offset, "size": size, "shape": shape}
        elif category.startswith('layer_'):
            layer_idx = int(category.split('_')[1])
            if layer_idx not in manifest['layers']:
                manifest['layers'][layer_idx] = {}
            manifest['layers'][layer_idx][san_name] = {"offset": off + data_offset, "size": size, "shape": shape}

    # Re-serialize manifest with corrected offsets
    manifest_json = json.dumps(manifest, indent=2)
    json_size = len(manifest_json.encode('utf-8'))

    # Write binary file with embedded header
    print(f"\nWriting {output_bin}...")
    offset = 0

    with open(output_bin, 'wb') as out_f:
        # Write header_size (uint32)
        out_f.write(struct.pack('<I', json_size))
        offset = 4

        # Write JSON manifest
        out_f.write(manifest_json.encode('utf-8'))
        offset += json_size

        # Pad to 64-byte alignment
        if offset % ALIGN != 0:
            pad = ALIGN - (offset % ALIGN)
            out_f.write(b'\x00' * pad)
            offset += pad

        # Write tensor data
        for san_name, orig_name, filename, off, size, shape, category in layout:
            filepath = model_path / filename
            header, data_start = header_cache[filename]

            # Read and write tensor data
            data, _, _ = read_tensor_data(str(filepath), header, data_start, orig_name)
            out_f.write(data)
            offset += size

    # Also write tensor_index.bin for fast C loading
    index_path = output_bin.parent / 'tensor_index.bin'
    print(f"\nWriting {index_path}...")
    with open(index_path, 'wb') as idx_f:
        idx_f.write(struct.pack('<I', 0x54504549))  # magic "IEPT"
        idx_f.write(struct.pack('<I', 1))            # version
        idx_f.write(struct.pack('<I', len(layout)))  # num_tensors
        idx_f.write(struct.pack('<Q', data_offset))  # data_start in model_weights.bin
        for san_name, orig_name, filename, off, size, shape, category in layout:
            encoded = san_name.encode('utf-8')
            idx_f.write(struct.pack('<I', len(encoded)))
            idx_f.write(encoded)
            idx_f.write(struct.pack('<Q', off + data_offset))
            idx_f.write(struct.pack('<Q', size))
    print(f"Tensor index written: {len(layout)} entries")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({total_bytes / elapsed / 1e9:.1f} GB/s)")

    # Summary
    print("\nExtracted tensors:")
    for cat in ['embed_tokens', 'lm_head', 'final_layer_norm']:
        if cat in manifest and manifest[cat]:
            info = manifest[cat]
            print(f"  {cat}: offset={info['offset']}, size={info['size']}, shape={info['shape']}")

    layer_count = len(manifest['layers'])
    print(f"  layers: {layer_count} layers")

    # Count expert tensors
    expert_tensors = 0
    for layer_idx, layer_data in manifest['layers'].items():
        if 'mlp' in layer_data:
            mlp = layer_data['mlp']
            if 'experts' in mlp:
                expert_tensors = len(mlp['experts'])
                break
    print(f"  experts per layer: {expert_tensors}")


if __name__ == '__main__':
    main()
