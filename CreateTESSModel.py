from transformers import AlbertModel, AlbertTokenizer
import torch
from dataclasses import dataclass
import os
from model_components import TESSTransformer


albert_model = 'model/albert_base_v2'


def create_tess_model(save_model_to, max_pos, albert_model):

    model = AlbertModel.from_pretrained(albert_model, local_files_only=True)
    tokenizer = AlbertTokenizer.from_pretrained(albert_model, local_files_only=True, model_max_length=max_pos)
    config = model.config
    config.num_hidden_groups = 1
    print(f"Albert Query Weight: {model.encoder.albert_layer_groups[0].albert_layers[0].attention.query.weight}",flush=True)

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos

    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        # new_pos_embed.weight.data[k:(k + step)] = model.embeddings.position_embeddings.weight.data[:]
        if k + step > max_pos:
            new_pos_embed[k:(max_pos)] = model.embeddings.position_embeddings.weight.data[:(max_pos - k)]
            k = max_pos
        else:
            new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight.data[:]
            k += step
    assert all(new_pos_embed[0] == new_pos_embed[512])
    assert all(new_pos_embed[255] == new_pos_embed[767])

    # Replace
    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # Replace the Encoder
    new_encoder = TESSTransformer(config)
    # Change encoder level parameters
    new_encoder.embedding_hidden_mapping_in = model.encoder.embedding_hidden_mapping_in
    # Change Layer Parameters
    albert_layer = model.encoder.albert_layer_groups[0].albert_layers[0]


    for group_id in range(config.num_hidden_groups):

        for layer_id in range(int(config.num_hidden_layers // config.num_hidden_groups)):

            new_encoder.tess_layer_groups[group_id].attentions[0].query = albert_layer.attention.query
            new_encoder.tess_layer_groups[group_id].attentions[0].key = albert_layer.attention.key
            new_encoder.tess_layer_groups[group_id].attentions[0].value = albert_layer.attention.value

            new_encoder.tess_layer_groups[group_id].attentions[0].attention_dropout = albert_layer.attention.attention_dropout
            new_encoder.tess_layer_groups[group_id].attentions[0].output_dropout = albert_layer.attention.output_dropout
            new_encoder.tess_layer_groups[group_id].attentions[0].dense  = albert_layer.attention.dense
            new_encoder.tess_layer_groups[group_id].attentions[0].LayerNorm  = albert_layer.attention.LayerNorm

            new_encoder.tess_layer_groups[group_id].ffn_group[layer_id].full_layer_layer_norm = albert_layer.full_layer_layer_norm
            new_encoder.tess_layer_groups[group_id].ffn_group[layer_id].ffn = albert_layer.ffn
            new_encoder.tess_layer_groups[group_id].ffn_group[layer_id].ffn_output = albert_layer.ffn_output
            new_encoder.tess_layer_groups[group_id].ffn_group[layer_id].activation = albert_layer.activation
            new_encoder.tess_layer_groups[group_id].ffn_group[layer_id].LayerNorm = albert_layer.attention.LayerNorm
            new_encoder.tess_layer_groups[group_id].ffn_group[layer_id].dropout = albert_layer.dropout

    # Change encoder
    new_layer = new_encoder.tess_layer_groups[0]
    assert len(new_layer.attentions) == 1
    assert new_layer.attentions[0].key.weight[0, 212] == albert_layer.attention.key.weight[0, 212]
    assert new_layer.attentions[0].dense.weight[0, 212] == albert_layer.attention.dense.weight[0, 212]
    assert new_layer.ffn_group[0].ffn.weight[28, 85] == albert_layer.ffn.weight[28, 85]
    assert new_layer.ffn_group[0].ffn.weight[28, 85] == albert_layer.ffn.weight[28, 85]

    model.encoder = new_encoder
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)

    return model, tokenizer




# ============= Evaluation: Only run if on main thread ==================#

if __name__ == '__main__':

    @dataclass
    class ModelArgs:
        attention_window = 768
        max_pos = 768

    model_args = ModelArgs()

    # AlbertLong Baseline
    model_path = f"model/tess_base_{model_args.max_pos}_shared"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model, tokenizer = create_tess_model(save_model_to=model_path,
                                         attention_window=model_args.attention_window,
                                         max_pos=model_args.max_pos,
                                         albert_model=albert_model)
    print(f'{model} Saved', flush = True)

