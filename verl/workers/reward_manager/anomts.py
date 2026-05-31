import torch
import numpy as np
from verl import DataProto
from collections import defaultdict
import json
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict
class AnomalyTimeSeriesReward:
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", max_resp_len=None, overlong_buffer_cfg=None,) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        self.compute_score = compute_score or _default_compute_score
        self.buffer = np.array((100,), dtype=bool)
        self.save_idx = 0
        self.reward_fn_key = reward_fn_key
    def print_format_success_rate(self):
        if self.save_idx >= 100:
            print("[Format success rate]: ", self.buffer.mean())
    def __call__(self, data: DataProto):

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        metrics_by_category = defaultdict(lambda: defaultdict(list))

        all_results_list = []
        
        already_print_data_sources = {}
        all_responses = []   


        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            response_ids = data_item.batch['responses']
            attention_mask = data_item.batch['attention_mask']

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = attention_mask[:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            valid_response_len = attention_mask[prompt_len:].sum()

            if valid_response_len == 0:
                continue
            valid_response_ids = response_ids[:valid_response_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info')
  
            category = extra_info.get('category', 'unknown')  
            image_path = extra_info.get('image_path', None)

            score, fmt, result_dict, pred_intervals, gt_intervals = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
      

            reward_tensor[i, valid_response_len - 1] = score
            

            current_result = {
                'affi precision': result_dict.get("affi precision", 0.0),
                'affi recall': result_dict.get("affi recall", 0.0),
                'affi f1': result_dict.get("affi f1", 0.0),
                'fmt_score': fmt,
                "class acc": result_dict.get("class_acc", 0.0),
            }
            all_results_list.append(current_result)
            
    
            for key, value in current_result.items():
                metrics_by_category[category][key].append(value)
            
 
            if category not in already_print_data_sources:
                already_print_data_sources[category] = 0
            each_category = max(2, int(self.num_examine / 8))
            if sum(already_print_data_sources.values()) < self.num_examine:
                already_print_data_sources[category] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
            all_responses.append(response_str) 

        overall_mean_metrics = {}
        if all_results_list:
   
            for key in all_results_list[0].keys():
                mean_value = np.mean([res[key] for res in all_results_list])
                overall_mean_metrics[f"mean_{key}"] = mean_value
        else: 
            overall_mean_metrics = {
                "mean_affinity_f1": 0.0, "mean_affinity_precision": 0.0,
                "mean_affinity_recall": 0.0, "mean_fmt_score": 0.0
            }


  
        return reward_tensor, overall_mean_metrics, None