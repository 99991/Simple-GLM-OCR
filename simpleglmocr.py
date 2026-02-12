import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import safetensors
from PIL import Image
import numpy as np
import regex
import json
import math
import os

class GlmOcrForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GlmOcrModel()
        self.lm_head = nn.Linear(1536, 59392, bias=False)

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None, attention_mask=None, cache_position=None, past_key_values=None, rope_deltas=None):
        hidden_states, rope_deltas = self.model(input_ids, pixel_values, image_grid_thw, cache_position, past_key_values, rope_deltas)
        logits = self.lm_head(hidden_states[:, -1, :])
        return logits, rope_deltas

class GlmOcrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = GlmOcrVisionModel()
        self.language_model = GlmOcrTextModel()
        self.spatial_merge_size = 2
        self.image_token_id = 59280

    def forward(self, input_ids, pixel_values, image_grid_thw, cache_position, past_key_values, rope_deltas):
        image_features = None
        if pixel_values is not None:
            image_features = self.visual(pixel_values.to(self.language_model.embed_tokens.weight.dtype), image_grid_thw)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if image_features is not None:
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw)
        else:
            q_len = input_ids.shape[1]
            position_ids = torch.arange(q_len, device=input_ids.device).view(1, -1) + (cache_position[0] + rope_deltas)
            position_ids = position_ids.unsqueeze(0).expand(3, 1, -1)

        return self.language_model(input_ids, inputs_embeds, position_ids, past_key_values), rope_deltas

    def get_rope_index(self, input_ids, image_grid_thw):
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id

        input_ids_list = input_ids[0].tolist()

        image_indices = [i for i, token in enumerate(input_ids_list) if token == image_token_id]

        if not image_indices:
            seq_len = len(input_ids_list)
            position_ids = torch.arange(seq_len, device=input_ids.device).view(1, -1).expand(3, -1).unsqueeze(1)
            mrope_position_deltas = torch.zeros((1, 1), device=input_ids.device, dtype=input_ids.dtype)
            return position_ids, mrope_position_deltas

        image_ranges = []
        if image_indices:
            start = image_indices[0]
            for i in range(1, len(image_indices)):
                if image_indices[i] != image_indices[i-1] + 1:
                    image_ranges.append((start, image_indices[i-1] + 1))
                    start = image_indices[i]
            image_ranges.append((start, image_indices[-1] + 1))

        llm_pos_ids_list = []
        current_pos = 0
        curr_llm_pos = 0
        image_index = 0

        for img_start, img_end in image_ranges:
            if img_start > current_pos:
                text_len = img_start - current_pos
                llm_pos_ids_list.append(torch.arange(curr_llm_pos, curr_llm_pos + text_len, device=input_ids.device).view(1, -1).expand(3, -1))
                curr_llm_pos += text_len

            t, h, w = image_grid_thw[image_index]
            llm_grid_t, llm_grid_h, llm_grid_w = t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size

            t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + curr_llm_pos)

            image_index += 1
            current_pos = img_end
            curr_llm_pos = llm_pos_ids_list[-1].max().item() + 1

        if current_pos < len(input_ids_list):
            text_len = len(input_ids_list) - current_pos
            llm_pos_ids_list.append(torch.arange(curr_llm_pos, curr_llm_pos + text_len, device=input_ids.device).view(1, -1).expand(3, -1))

        position_ids = torch.cat(llm_pos_ids_list, dim=1).unsqueeze(1)
        mrope_position_deltas = (position_ids[..., -1].max() + 1 - len(input_ids_list)).view(1, 1)

        return position_ids, mrope_position_deltas

class GlmOcrVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = GlmOcrVisionPatchEmbed()
        self.blocks = nn.ModuleList([GlmOcrVisionBlock() for _ in range(24)])
        self.merger = GlmOcrVisionPatchMerger()
        self.downsample = nn.Conv2d(1024, 1536, kernel_size=(2, 2), stride=(2, 2))
        self.post_layernorm = GlmOcrRMSNorm((1024,), eps=1e-05)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_emb_module(max_grid_size, device=grid_thw.device)
        return rotary_pos_emb_full[pos_ids].to(grid_thw.device).flatten(1), pos_ids

    def rotary_emb_module(self, seqlen, device):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 32, 2, device=device).float() / 32))
        seq = torch.arange(seqlen, device=device).float()
        return torch.outer(seq, inv_freq)

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb, _ = self.rot_pos_emb(grid_thw)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        pos_emb_vision = (emb.cos(), emb.sin())
        for block in self.blocks:
            hidden_states = block(hidden_states, pos_emb_vision)

        image_features = self.post_layernorm(hidden_states)
        image_features = image_features.view(-1, 2, 2, image_features.shape[-1]).permute(0, 3, 1, 2)
        image_features = self.downsample(image_features).view(-1, 1536)
        image_features = self.merger(image_features)
        return image_features

class GlmOcrVisionPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=14, temporal_patch_size=2, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=(temporal_patch_size, patch_size, patch_size), stride=(temporal_patch_size, patch_size, patch_size))

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class GlmOcrVisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = GlmOcrRMSNorm((1024,), eps=1e-05)
        self.norm2 = GlmOcrRMSNorm((1024,), eps=1e-05)
        self.attn = GlmOcrVisionAttention()
        self.mlp = GlmOcrVisionMlp()

    def forward(self, x, position_embeddings):
        x = x + self.attn(self.norm1(x), position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    cos = cos.to(q.device)
    sin = sin.to(k.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

class GlmOcrVisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.head_dim = 64
        self.qkv = nn.Linear(1024, 3072, bias=True)
        self.proj = nn.Linear(1024, 1024, bias=True)
        self.q_norm = GlmOcrRMSNorm((64,), eps=1e-05)
        self.k_norm = GlmOcrRMSNorm((64,), eps=1e-05)

    def forward(self, hidden_states, position_embeddings):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class GlmOcrRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class GlmOcrVisionMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(1024, 4096, bias=True)
        self.up_proj = nn.Linear(1024, 4096, bias=True)
        self.down_proj = nn.Linear(4096, 1024, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class GlmOcrVisionPatchMerger(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1536, 1536, bias=False)
        self.post_projection_norm = nn.LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        self.gate_proj = nn.Linear(1536, 4608, bias=False)
        self.up_proj = nn.Linear(1536, 4608, bias=False)
        self.down_proj = nn.Linear(4608, 1536, bias=False)
        self.act1 = nn.GELU()
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class GlmOcrTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(59392, 1536, padding_idx=59246)
        self.layers = nn.ModuleList([GlmOcrTextDecoderLayer(i) for i in range(16)])
        self.norm = GlmOcrRMSNorm((1536,), eps=1e-05)
        self.rotary_emb = GlmOcrTextRotaryEmbedding()

    def forward(self, input_ids, inputs_embeds, position_ids, past_key_values):
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, past_key_values)

        return self.norm(hidden_states)

class GlmOcrTextDecoderLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = GlmOcrTextAttention(layer_idx)
        self.mlp = GlmOcrTextMLP()
        self.input_layernorm = GlmOcrRMSNorm((1536,), eps=1e-05)
        self.post_attention_layernorm = GlmOcrRMSNorm((1536,), eps=1e-05)
        self.post_self_attn_layernorm = GlmOcrRMSNorm((1536,), eps=1e-05)
        self.post_mlp_layernorm = GlmOcrRMSNorm((1536,), eps=1e-05)

    def forward(self, hidden_states, position_embeddings, past_key_values):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, position_embeddings, past_key_values)
        hidden_states = residual + self.post_self_attn_layernorm(attn_out)

        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        x = self.post_mlp_layernorm(x)
        return hidden_states + x

def append_kv_cache(cache, idx, key, value):
    entry = cache.get(idx)

    if entry is None:
        cache[idx] = {"key": key, "value": value}
    else:
        key = torch.cat([entry["key"], key], dim=-2)
        value = torch.cat([entry["value"], value], dim=-2)

        entry["key"] = key
        entry["value"] = value

    return key, value

class GlmOcrTextAttention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.num_heads = 16
        self.head_dim = 128
        self.num_key_value_heads = 8
        self.q_proj = nn.Linear(1536, 2048, bias=False)
        self.k_proj = nn.Linear(1536, 1024, bias=False)
        self.v_proj = nn.Linear(1536, 1024, bias=False)
        self.o_proj = nn.Linear(2048, 1536, bias=False)
        self.layer_idx = layer_idx
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, position_embeddings, past_key_values):
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key, value = append_kv_cache(past_key_values, self.layer_idx, key, value)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, is_causal=query.shape[2] > 1, scale=self.scaling, enable_gqa=True
        ).transpose(1, 2).reshape(bsz, q_len, -1)

        return self.o_proj(attn_output)


class GlmOcrTextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(1536, 9216, bias=False)
        self.down_proj = nn.Linear(4608, 1536, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(up_states * F.silu(gate))

class GlmOcrTextRotaryEmbedding(nn.Module):
    def forward(self, x, position_ids):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 128, 2, device=x.device).float() / 128))
        freqs = (inv_freq[None, None, :, None] @ position_ids[:, :, None, :].float()).transpose(2, 3)

        c1, c2, c3 = freqs.split([16, 24, 24], dim=-1)
        freqs = torch.cat([c1[0], c2[1], c3[2]], dim=-1)

        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

def rotate_half_llm(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)

def load_model():
    filename = os.path.expanduser("~/.cache/huggingface/hub/models--zai-org--GLM-OCR/blobs/a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815")

    if not os.path.exists(filename):
        url = "https://huggingface.co/zai-org/GLM-OCR/resolve/main/model.safetensors"
        filename = download(url)

    f = safetensors.safe_open(filename, framework="pt", device="cpu")

    d = {}
    for key in f.keys():
        # Discard unused layer, no idea why it is even in there
        if not key.startswith("model.language_model.layers.16."):
            value = f.get_tensor(key)
            d[key] = value

    # Creating model with meta device skips costly initialization of weights, which would be overwritten by load_state_dict anyway
    with torch.device("meta"):
        model = GlmOcrForConditionalGeneration()

    model.load_state_dict(d, assign=True)

    return model

# NB: Tokenizer and processor are mostly vibe-coded. Don't read too much into it.
class GlmOcrTokenizer:
    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data['model']['vocab']
        self.special_tokens = {}
        if 'added_tokens' in data:
            for token_data in data['added_tokens']:
                content = token_data['content']
                self.vocab[content] = token_data['id']
                self.special_tokens[content] = token_data['id']

        self.id_to_token = {v: k for k, v in self.vocab.items()}

        merges = data['model']['merges']
        self.merges = {}
        for i, m in enumerate(merges):
            pair = tuple(m.split()) if isinstance(m, str) else tuple(m)
            self.merges[pair] = i

        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Identify special tokens for regex splitting
        special_pat = "|".join(regex.escape(k) for k in sorted(list(self.special_tokens.keys()), key=len, reverse=True))

        # Regex pattern from Glm4
        self.pat = regex.compile(f"({special_pat})|" + r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

        self.cache = {}

    def bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if bigram not in self.merges:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def tokenize(self, text):
        bpe_tokens = []
        for match in self.pat.finditer(text):
            if match.group(1):
                bpe_tokens.append(match.group(1))
            else:
                whole = match.group()
                token = "".join(self.byte_encoder[b] for b in whole.encode("utf-8"))
                bpe_tokens.extend(self.bpe(token).split(" "))
        return bpe_tokens

    def encode(self, text):
        # Sort special tokens by length descending
        special_tokens = sorted(self.special_tokens.keys(), key=len, reverse=True)

        # Split text into parts, keeping special tokens as separate items
        if not special_tokens:
            parts = [text]
        else:
            special_re = regex.compile("(" + "|".join(regex.escape(st) for st in special_tokens) + ")")
            parts = []
            last_idx = 0
            for match in special_re.finditer(text):
                start, end = match.span()
                if start > last_idx:
                    parts.append(text[last_idx:start])
                parts.append(("SPECIAL", match.group()))
                last_idx = end
            if last_idx < len(text):
                parts.append(text[last_idx:])

        ids = []
        for p in parts:
            if isinstance(p, tuple):
                ids.append(self.vocab[p[1]])
            else:
                # Use pat without special tokens for the text parts
                # Actually, self.pat still has them but they won't match in p
                for match in self.pat.finditer(p):
                    whole = match.group()
                    token = "".join(self.byte_encoder[b] for b in whole.encode("utf-8"))
                    bpe_tokens = self.bpe(token).split(" ")
                    ids.extend([self.vocab[t] for t in bpe_tokens])
        return ids

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()

        result_tokens = []
        byte_data = bytearray()

        for i in ids:
            token = self.id_to_token[i]
            if token in self.special_tokens:
                if byte_data:
                    result_tokens.append(byte_data.decode("utf-8", errors="replace"))
                    byte_data = bytearray()
                result_tokens.append(token)
            else:
                for char in token:
                    if char in self.byte_decoder:
                        byte_data.append(self.byte_decoder[char])
                    else:
                        if byte_data:
                            result_tokens.append(byte_data.decode("utf-8", errors="replace"))
                            byte_data = bytearray()
                        result_tokens.append(char)
        if byte_data:
            result_tokens.append(byte_data.decode("utf-8", errors="replace"))
        return "".join(result_tokens)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens)
        return [self.vocab.get(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return [self.id_to_token.get(i) for i in ids]

class GlmOcrProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.config = {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "shortest_edge": 112 * 112,
            "longest_edge": 9633792
        }

    def apply_chat_template(self, messages, add_generation_prompt=False):
        prompt = "[gMASK]<sop>"
        for m in messages:
            role = m['role']
            content = m['content']
            prompt += f"<|{role}|>\n"
            if isinstance(content, list):
                for item in content:
                    if item['type'] == 'image':
                        prompt += "<|begin_of_image|><|image|><|end_of_image|>"
                    elif item['type'] == 'text':
                        prompt += item['text']
            else:
                prompt += content
        if add_generation_prompt:
            prompt += "<|assistant|>\n"
        return prompt

    def smart_resize(self, num_frames, height, width, temporal_factor=2, factor=28, min_pixels=112*112, max_pixels=9633792):
        if height < factor or width < factor:
            scale = max(factor / height, factor / width)
            height = int(height * scale)
            width = int(width * scale)

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        t_bar = round(num_frames / temporal_factor) * temporal_factor

        if t_bar * h_bar * w_bar > max_pixels:
            beta = math.sqrt((num_frames * height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif t_bar * h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (num_frames * height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        num_frames = self.config['temporal_patch_size']
        width, height = image.size
        factor = self.config['patch_size'] * self.config['merge_size']

        h_bar, w_bar = self.smart_resize(num_frames, height, width, factor=factor,
                                          min_pixels=self.config['shortest_edge'],
                                          max_pixels=self.config['longest_edge'])

        image = image.resize((w_bar, h_bar), Image.BICUBIC)
        data = np.array(image).astype(np.float32) / 255.0
        data = (data - self.config['image_mean']) / self.config['image_std']

        # (H, W, C) -> (C, H, W)
        data = data.transpose(2, 0, 1)
        # Add T dimension: (T, C, H, W)
        data = data[np.newaxis, ...]

        # Pad temporal to multiple of temporal_patch_size
        tp = self.config['temporal_patch_size']
        if data.shape[0] % tp != 0:
            repeats = np.tile(data[-1:], (tp - (data.shape[0] % tp), 1, 1, 1))
            data = np.concatenate([data, repeats], axis=0)

        t_len, channel, h, w = data.shape
        grid_t = t_len // tp
        ps = self.config['patch_size']
        ms = self.config['merge_size']
        grid_h, grid_w = h // ps, w // ps

        # Reshape to (grid_t, tp, C, gh//ms, ms, ps, gw//ms, ms, ps)
        data = data.reshape(grid_t, tp, channel, grid_h // ms, ms, ps, grid_w // ms, ms, ps)
        # Transpose to (grid_t, gh//ms, gw//ms, ms, ms, C, tp, ps, ps)
        data = data.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        data = data.reshape(-1, channel * tp * ps * ps)

        return torch.from_numpy(data), (grid_t, grid_h, grid_w)

    def __call__(self, text, images=None, add_generation_prompt=False):
        if isinstance(text, list):
            prompt = self.apply_chat_template(text, add_generation_prompt)
        else:
            prompt = text

        pixel_values = None
        image_grid_thw = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            pixel_values_list = []
            grid_thw_list = []
            for img in images:
                pv, thw = self.preprocess_image(img)
                pixel_values_list.append(pv)
                grid_thw_list.append(thw)

            ms = self.config['merge_size']
            image_token = "<|image|>"
            for i, thw in enumerate(grid_thw_list):
                num_tokens = (thw[0] * thw[1] * thw[2]) // (ms**2)
                prompt = prompt.replace(image_token, "<|placeholder|>" * num_tokens, 1)
            prompt = prompt.replace("<|placeholder|>", image_token)

            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_grid_thw = torch.tensor(grid_thw_list)

        input_ids = self.tokenizer.encode(prompt)
        return {
            "input_ids": torch.tensor([input_ids]),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long)
        }


def load_tokenizer():
    filename = os.path.expanduser("~/.cache/huggingface/hub/models--zai-org--GLM-OCR/blobs/9f4a549a14a96217569648aa7627c6674ad94fe9")

    if not os.path.exists(filename):
        url = "https://huggingface.co/zai-org/GLM-OCR/resolve/main/tokenizer.json"
        filename = download(url)

    return GlmOcrTokenizer(filename)

def download(url):
    filename = url.rsplit("/", 1)[-1]
    if os.path.exists(filename):
        return filename

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, 100 * downloaded // total_size)
        bar = "█" * (percent // 2) + "-" * (50 - percent // 2)
        print(f"\r{filename}  |{bar}| {percent:3}%", flush=True, end="")

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, filename, reporthook=_progress)
    print()          # final newline
    return filename

class SimpleGlmOcr:
    def __init__(self, device=None):
        tokenizer = load_tokenizer()
        processor = GlmOcrProcessor(tokenizer)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model().to(device)
        model.dtype = torch.bfloat16
        model.device = device

        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model

    def generate_tokens_logits(self, prompt, image, max_tokens=8192):
        model = self.model
        device = model.device

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        custom_inputs = self.processor(messages, images=image, add_generation_prompt=True)
        custom_inputs = {k: v.to(model.device) if v is not None else None for k, v in custom_inputs.items()}

        input_ids = custom_inputs["input_ids"]
        pixel_values = custom_inputs["pixel_values"]
        image_grid_thw = custom_inputs["image_grid_thw"]
        attention_mask = custom_inputs["attention_mask"]
        seq_len = input_ids.shape[1]

        # Decode and print output
        with torch.no_grad():
            generated_ids = []
            curr_past_key_values = {}
            curr_input_ids = input_ids
            curr_pixel_values = pixel_values
            curr_image_grid_thw = image_grid_thw
            curr_attention_mask = attention_mask
            rope_deltas = None
            cache_position = torch.arange(input_ids.shape[1], device=device)

            # NB: Actual <|endoftext|> is 59246 according to tokenizer.json, but model always ends with <|user|>: 59253 instead for some reason, so we use that instead
            eos_token = 59253

            for step in range(max_tokens):
                logits, rope_deltas = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=curr_attention_mask,
                    cache_position=cache_position,
                    past_key_values=curr_past_key_values,
                    rope_deltas=rope_deltas,
                )

                next_token_id = torch.argmax(logits[0]).item()

                if next_token_id == eos_token:
                    break

                yield next_token_id, logits

                curr_input_ids = torch.cat([curr_input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
                curr_attention_mask = torch.cat([curr_attention_mask, torch.ones((1, 1), device=device)], dim=1)
                pixel_values = None
                image_grid_thw = None
                input_ids = curr_input_ids[:, -1:]
                cache_position = torch.tensor([curr_input_ids.shape[1] - 1], device=device)

    def run(self, prompt, image, max_tokens=8192):
        tokens = []
        for token, _ in self.generate_tokens_logits(prompt, image, max_tokens=max_tokens):
            tokens.append(token)
        return self.tokenizer.decode(tokens)
