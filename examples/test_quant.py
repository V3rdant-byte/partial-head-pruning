import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
import random
from torch.nn.functional import pad

#import sys
#sys.path.insert(1, '../../models/opt-1.3b/')
#from modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM

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
            #if layer < 23:
            #    less_important_indices_layers[layer + 1] = torch.cat(
            #    (less_important_indices_layers[layer + 1], less_important_indices_layers[layer]), -1)
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
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            output = outputs
            last_token_logits = output.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

    @torch.no_grad()
    def evaluate_with_latency(self, model, top_k=16):
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
                    multihead_sum_array = torch.sum(torch.abs(head_matrix_tensor), (3, 2))
                    sorted_ascending_head_array, sorted_indices = torch.sort(multihead_sum_array)

                    layer_head_sum_ascend_list.append(sorted_ascending_head_array)
                    layer_head_sum_ascend_indices.append(sorted_indices[0][0:top_k])
                attention_heads_list.append(outputs.attentions)
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            count += 1

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy, attention_heads_list, layer_head_sum_ascend_list, layer_head_sum_ascend_indices

    @torch.no_grad()
    def evaluate_with_quantization_for_all_batches(self, model, top_k=1):
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
    def evaluate_with_quantization_for_first_batch_random(self, model, top_k=16):
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
            multihead_sum_array_across_layers = torch.zeros(1, 32).to(self.device)
            multihead_sum_array_across_layers_each_layer = torch.zeros(24, 32).to(self.device)
            if count == 0:
                for layer in range(len(outputs.attentions)):
                    head_matrix_tensor = outputs.attentions[layer]
                    # reduce in the head_dim and target_len dimension
                    multihead_sum_array = torch.sum(torch.abs(head_matrix_tensor), (3, 2)) # 1 * 32 * 512 * 64
                    multihead_sum_array_across_layers += multihead_sum_array
                    sorted_ascending_head_array, sorted_indices = torch.sort(multihead_sum_array_across_layers)

                    layer_head_sum_ascend_list.append(sorted_ascending_head_array)
                    #layer_head_sum_ascend_indices.append(sorted_indices[0][0:top_k])
                    layer_head_sum_ascend_indices.append(torch.randint(0, 32, (1, top_k))[0])
                    if layer != 0:
                        layer_head_sum_ascend_indices[layer] = torch.unique(torch.cat((layer_head_sum_ascend_indices[layer-1], layer_head_sum_ascend_indices[layer]), -1))
                    multihead_sum_array_across_layers_each_layer[layer] = multihead_sum_array_across_layers
                attention_heads_list.append(multihead_sum_array_across_layers_each_layer)
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
            multihead_sum_array_across_layers = torch.zeros(1, 32).to(self.device)
            multihead_sum_array_across_layers_each_layer = torch.zeros(24, 32).to(self.device)
            if count == 0:
                for layer in range(len(outputs.attentions)):
                    head_matrix_tensor = outputs.attentions[layer]
                    # reduce in the head_dim and target_len dimension
                    multihead_sum_array = torch.sum(torch.abs(head_matrix_tensor), (3, 2)) # 1 * 32 * 512 * 64
                    multihead_sum_array_across_layers += multihead_sum_array
                    sorted_ascending_head_array, sorted_indices = torch.sort(multihead_sum_array_across_layers)

                    layer_head_sum_ascend_list.append(sorted_ascending_head_array)
                    layer_head_sum_ascend_indices.append(sorted_indices[0][0:top_k])
                    #layer_head_sum_ascend_indices.append(torch.tensor(random.randint(0, 31)).reshape(1))
                    if layer != 0:
                        layer_head_sum_ascend_indices[layer] = torch.unique(torch.cat((layer_head_sum_ascend_indices[layer-1], layer_head_sum_ascend_indices[layer]), -1))
                    multihead_sum_array_across_layers_each_layer[layer] = multihead_sum_array_across_layers
                attention_heads_list.append(multihead_sum_array_across_layers_each_layer)
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

def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


from datasets import load_dataset

def print_sm():
    # q_proj 2048 * 2048
    # multi head matrix 32 * 512 * 64
    act_scales = torch.load('act_scales/opt-1.3b.pt')
    smooth_lm(model_fp16, act_scales, 0.5)
    model_smoothquant_w8a8 = quantize_model(model_fp16)
    acc2_smoothquant_w8a8, latency_smoothquant_w8a8, attention_heads_list_sm88, layer_head_sum_ascend_list_sm88, layer_head_sum_ascend_indices_sm88 = evaluator.evaluate_with_latency(
        model_smoothquant_w8a8)
    # for i in range(len(attention_heads_list_sm88)):
    # print(attention_heads_list_sm88[i])
    print(f'SmoothQuant INT8 accuracy: {acc2_smoothquant_w8a8}, per-sample lantecy: {latency_smoothquant_w8a8:.3f}ms')

tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-1.3b')
# ../../../.cache/huggingface/modules/datasets_modules/datasets/lambada/2e4879aaaa342d8f748b7275991006d2e27a8b0abc0a28ea299b3e3b839a3a40
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer, 'cuda')

model_fp16 = OPTForCausalLM.from_pretrained(
    'facebook/opt-1.3b', torch_dtype=torch.float16, device_map='auto', output_attentions=True)

act_scales = torch.load('act_scales/opt-1.3b.pt')
smooth_lm(model_fp16, act_scales, 0.5)
acc2_fp16_4, latency_fp16, attention_heads_list_fp16, layer_head_sum_ascend_list_fp16, layer_head_sum_ascend_indices_fp16, model_w8a8 = evaluator.evaluate_with_quantization_for_first_batch(model_fp16, 8)
print(f'fp16 accuracy: {acc2_fp16_4:.4f}, per-sample lantecy: {latency_fp16:.3f}ms')

model_fp16 = OPTForCausalLM.from_pretrained(
    'facebook/opt-1.3b', torch_dtype=torch.float16, device_map='auto', output_attentions=True)

act_scales = torch.load('act_scales/opt-1.3b.pt')
smooth_lm(model_fp16, act_scales, 0.5)
acc2_fp16_24, latency_fp16, a, b, c, d = evaluator.evaluate_with_quantization_for_first_batch(model_fp16, 24)
print(f'fp16 accuracy: {acc2_fp16_24:.4f}, per-sample lantecy: {latency_fp16:.3f}ms')
#model_w8a8 = quantize_model(model_fp16)
#print_model_size(model_w8a8)
#acc2_fp16, latency_fp16, attention_heads_list_fp16, layer_head_sum_ascend_list_fp16, layer_head_sum_ascend_indices_fp16 = evaluator.evaluate_with_latency(model_w8a8)
#file1 = open("cumulated heads across layers.txt", "w")
#print(attention_heads_list_fp16[0].size())
#for i in range(len(attention_heads_list_fp16[0])):
#    print(attention_heads_list_fp16[0][i].cpu().numpy())
#    file1.write(str(attention_heads_list_fp16[0][i].cpu().numpy()))
#    file1.write('\n')


   # print(layer_head_sum_ascend_list_fp16)               # sum of head values list in ascending order
    #print(layer_head_sum_ascend_indices_fp16)            # indices of the heads of the list above

    #print(len(layer_head_sum_ascend_indices_fp16))       # there are a total of 24 layers
    #print(layer_head_sum_ascend_indices_fp16[0].size())  # size of one layer = 32 attention heads

#file1.close()
# k_proj: (2048, 2048) -> k_proj(hidden_state) : (1, 512, 2048) -> _shape(k_proj(hidden_state), -1, 1) : (1, 32, 512, 64)
#print_sm()
















#model_w8a8 = quantize_model(model_fp16)
#acc_w8a8 = evaluator.evaluate(model_w8a8)
#acc2_w8a8, latency_w8a8, a, b, c = evaluator.evaluate_with_latency(model_w8a8)
#print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')
#print(f'W8A8 accuracy: {acc2_w8a8:4f}, per-sample lantecy: {latency_w8a8:.3f}ms')



#model = OPTForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16, device_map='auto')
#act_scales = torch.load('act_scales/opt-1.3b.pt')
#smooth_lm(model_fp16, act_scales, 0.5)
#model_smoothquant_w8a8 = quantize_model(model_fp16)
#acc2_w8a8, latency_w8a8, a, b, c = evaluator.evaluate_with_latency(model_smoothquant_w8a8)
#print(f'Naive W8A8 quantized model accuracy: {acc2_w8a8}')
#print(f'W8A8 accuracy: {acc2_w8a8:4f}, per-sample lantecy: {latency_w8a8:.3f}ms')
#print(model_smoothquant_w8a8)
#acc_randomly_select_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
#acc2_randomly_select_smoothquant_w8a8, latency_randomly_select_smoothquant_w8a8 = evaluator.evaluate_with_latency(model_smoothquant_w8a8)
#print(f'SmoothQuant W8A8 randomly select quantized model accuracy: {acc_randomly_select_smoothquant_w8a8}')
#print(f'SmoothQuant INT8 randomly select accuracy: {acc2_randomly_select_smoothquant_w8a8}, per-sample lantecy: {latency_randomly_select_smoothquant_w8a8:.3f}ms')