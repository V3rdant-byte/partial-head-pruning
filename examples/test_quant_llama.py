import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
)
from transformers import LlamaTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_llama_like
import tqdm
def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    less_important_indices_layers = [0]
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, less_important_indices_layers, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, less_important_indices_layers, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, less_important_indices_layers, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, less_important_indices_layers, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, less_important_indices_layers, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, less_important_indices_layers, weight_quant=weight_quant, act_quant=act_quant)
    return model

def quantize_model_less_important_head(model, less_important_indices_layers, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    layer = 0
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, less_important_indices_layers[layer], weight_quant='per_head', act_quant='per_head')
            m.fc2 = W8A8Linear.from_float(m.fc2, less_important_indices_layers[layer], weight_quant='per_head', act_quant='per_head')
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            #print(less_important_indices_layers[layer])
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, less_important_indices_layers[layer], weight_quant='per_head', act_quant='per_head', quantize_output=quantize_bmm_input, is_attention=True)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, less_important_indices_layers[layer], weight_quant='per_head', act_quant='per_head', quantize_output=quantize_bmm_input, is_attention=True)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, less_important_indices_layers[layer], weight_quant='per_head', act_quant='per_head', quantize_output=quantize_bmm_input, is_attention=True)
            m.out_proj = W8A8Linear.from_float(m.out_proj, less_important_indices_layers[layer], weight_quant=weight_quant, act_quant='per_head', is_attention=True)
            layer = layer + 1
    return model

def quantize_model_randomly_selected(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True, output_attentions=False):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            do_quantize = random.randint(0, 1)
            if do_quantize:
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

    return model
class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits, self_atten_weight = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048)), self_atten_weight

    @torch.no_grad()
    def evaluate_with_quantization_for_all_batches(self, model, top_k=2):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        count = 0
        attention_heads_list = []
        layer_head_sum_ascend_list = []
        layer_head_sum_ascend_indices = []
        for batch in self.dataset:

            #layer_head_sum_ascend_list = []
            layer_head_sum_ascend_indices_per_batch = []
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            #if count == 0:
            for layer in range(len(outputs.attentions)):
                head_matrix_tensor = outputs.attentions[layer]
                # reduce in the head_dim and target_len dimension
                multihead_sum_array = torch.sum(torch.abs(head_matrix_tensor), (3, 2))
                sorted_ascending_head_array, sorted_indices = torch.sort(multihead_sum_array)

                #layer_head_sum_ascend_list.append(sorted_ascending_head_array)
                layer_head_sum_ascend_indices_per_batch.append(sorted_indices[0][0:top_k])
            attention_heads_list.append(outputs.attentions)
            model = quantize_model_less_important_head(model, layer_head_sum_ascend_indices_per_batch)
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            count += 1
            layer_head_sum_ascend_indices.append(layer_head_sum_ascend_indices_per_batch)

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy, attention_heads_list, layer_head_sum_ascend_list, layer_head_sum_ascend_indices, model

    @torch.no_grad()
    def evaluate_with_quantization_for_first_batch(self, model, top_k=16):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        count = 0
        attention_heads_list = []
        layer_head_sum_ascend_list = []
        layer_head_sum_ascend_indices = []
        for batch in self.dataset:

            #layer_head_sum_ascend_list = []
            #layer_head_sum_ascend_indices_per_batch = []
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            if count == 0:
                for layer in range(len(outputs.attentions)):
                    head_matrix_tensor = outputs.attentions[layer]
                    # reduce in the head_dim and target_len dimension
                    multihead_sum_array = torch.sum(torch.abs(head_matrix_tensor), (3, 2)) # 1 * 32 * 512 * 64
                    sorted_ascending_head_array, sorted_indices = torch.sort(multihead_sum_array)

                    layer_head_sum_ascend_list.append(sorted_ascending_head_array)
                    layer_head_sum_ascend_indices.append(sorted_indices[0][0:top_k])
                attention_heads_list.append(outputs.attentions)
                model = quantize_model_less_important_head(model, layer_head_sum_ascend_indices)
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            count += 1

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy, attention_heads_list, layer_head_sum_ascend_list, layer_head_sum_ascend_indices, model

from datasets import load_dataset

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
evaluator = Evaluator(dataset, tokenizer, "cuda")

model_fp16 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto", output_attentions=True
)
ppl_fp16 = evaluator.evaluate(model_fp16)
print(f"Original model (fp16) perplexity: {ppl_fp16}")

#act_scales = torch.load('act_scales/opt-1.3b.pt')
#smooth_lm(model_fp16, act_scales, 0.5)
#acc2_fp16, latency_fp16, attention_heads_list_fp16, layer_head_sum_ascend_list_fp16, layer_head_sum_ascend_indices_fp16, model_w8a8 = evaluator.evaluate_with_quantization_for_first_batch(model_fp16)
#model_w8a8 = quantize_model(model_fp16)
#print_model_size(model_w8a8)
#acc2_fp16, latency_fp16, attention_heads_list_fp16, layer_head_sum_ascend_list_fp16, layer_head_sum_ascend_indices_fp16 = evaluator.evaluate_with_latency(model_w8a8)

#for i in range(len(attention_heads_list_fp16)):
 #   print(attention_heads_list_fp16[i])
  #  print(attention_heads_list_fp16[i][0].size())

   # print(layer_head_sum_ascend_list_fp16)               # sum of head values list in ascending order
    #print(layer_head_sum_ascend_indices_fp16)            # indices of the heads of the list above

    #print(len(layer_head_sum_ascend_indices_fp16))       # there are a total of 24 layers
    #print(layer_head_sum_ascend_indices_fp16[0].size())  # size of one layer = 32 attention heads
print(f'fp16 accuracy: {acc2_fp16}, per-sample lantecy: {latency_fp16:.3f}ms')

# k_proj: (2048, 2048) -> k_proj(hidden_state) : (1, 512, 2048) -> _shape(k_proj(hidden_state), -1, 1) : (1, 32, 512, 64)
#print_sm()
















#model_w8a8 = quantize_model(model_fp16)
#acc_w8a8 = evaluator.evaluate(model_w8a8)
#acc2_w8a8, latency_w8a8 = evaluator.evaluate_with_latency(model_w8a8)
#print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')
#print(f'W8A8 accuracy: {acc2_w8a8}, per-sample lantecy: {latency_w8a8:.3f}ms')



#model = OPTForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16, device_map='auto')
#act_scales = torch.load('act_scales/opt-1.3b.pt')
#smooth_lm(model, act_scales, 0.5)
#model_smoothquant_w8a8 = quantize_model_randomly_selected(model)
#print(model_smoothquant_w8a8)
#acc_randomly_select_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
#acc2_randomly_select_smoothquant_w8a8, latency_randomly_select_smoothquant_w8a8 = evaluator.evaluate_with_latency(model_smoothquant_w8a8)
#print(f'SmoothQuant W8A8 randomly select quantized model accuracy: {acc_randomly_select_smoothquant_w8a8}')
#print(f'SmoothQuant INT8 randomly select accuracy: {acc2_randomly_select_smoothquant_w8a8}, per-sample lantecy: {latency_randomly_select_smoothquant_w8a8:.3f}ms')