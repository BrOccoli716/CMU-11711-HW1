# 11711 ANLP Assignment 1 Report

### Yiming Fu (Andrew ID: yimingfu)

#### Summary

In this assignment, I have chosen the second provided option for A+ improvement, which is 

- Enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.

Specifically, I have realized **padding mask** and **causal mask** for our Llama model, and achieved significant accuracy improvements in the **zero-shot prompting** setting.

*Note:* The padding mask and causal mask mechanism is added to the Llama model's forward function. But there is no training process in the zero-shot prompting setting, so the newly-added mechanism only affects the inference process.



#### Code Explanation



#### Result Display

**SST Zero-Shot Prompting**

The original result without padding mask is shown below.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925133736131.png' width=70%>

The advanced result with padding mask is shown below.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925133553117.png' width=70%>

Accuracies on both the validation and test dataset has increased significantly.

**CFIMDB Zero-Shot Prompting**

The original result without padding mask is shown below.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925134800132.png' width=70%>

The advanced result with padding mask is shown below.

<img src='/Users/fuyiming/Library/Application Support/typora-user-images/image-20250925132947332.png' width=70%>

Accuracies on both the validation and test dataset has increased significantly.



#### Executing Instructions

For the **SST Zero-Shot Prompting** scenario, run the following shell script.

```shell
python run_llama_new.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-advanced-output.txt --test_out sst-test-advanced-output.txt --use_gpu
```

For the **CFIMDB Zero-Shot Prompting** scenario, run the following shell script.

```shell
python run_llama_new.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-advanced-output.txt --test_out cfimdb-test-advanced-output.txt --use_gpu
```

