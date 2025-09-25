# 11711 ANLP Assignment 1 Report

### Yiming Fu (Andrew ID: yimingfu)

#### Summary

In this assignment, I have chosen the second provided option for A+ improvement, which is 

- Enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.

Specifically, I have realized **padding mask** and **causal mask** for our Llama model, and achieved significant accuracy improvements in the **zero-shot prompting** setting.

*Note:* The padding mask and causal mask mechanism is added to the Llama model's forward function. But there is no training process in the zero-shot prompting setting, so the newly-added mechanism only affects the inference process.

#### Code Explanation

In `llama.py`, we add the padding mask and causal mask for "compute_query_key_value_scores" function in the "Attention" class, which includes these two masks in attention calculation.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925155949338.png' width=60%>

Then in the "Llama" class, we modify the logic to fetch the last **effective** token, which is neither padding nor null. (The token before the first padding is null, so we select the token before it.)

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925160322759.png' width=60%>

Finally, we create a `run_llama_new.py` file, which is quite similar to `run_llama.py`. Here we need to return padding masks in "LlamaDataset", and modify corresponding train & eval procedures.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925160901269.png' width=60%>

#### Result Display

**SST Zero-Shot Prompting**

The advanced result with padding mask is shown below.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925133553117.png' width=50%>

Accuracies on both dev and test dataset have increased significantly. (0.237/0.250 before)

**CFIMDB Zero-Shot Prompting**

The advanced result with padding mask is shown below.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925132947332.png' width=50%>

Accuracies on both dev and test dataset have increased significantly. (0.490/0.109 before)

#### Executing Instructions

For **SST Zero-Shot Prompting**, run the following shell script to execute `run_llama_new.py`.

```shell
python run_llama_new.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-advanced-output.txt --test_out sst-test-advanced-output.txt --use_gpu
```

For **CFIMDB Zero-Shot Prompting**, run the following shell script to execute `run_llama_new.py`.

```shell
python run_llama_new.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-advanced-output.txt --test_out cfimdb-test-advanced-output.txt --use_gpu
```

