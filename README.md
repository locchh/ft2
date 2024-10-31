# ft2
fine tune llama 3.2

<div align="center">
  <img src="assets/appoarch.png" width="500">
</div>

- train command: 
```
nohup python train_full.py --data_file data/sarcasm.csv \
--output_dir logs \
--num_train_epochs 3 \
--batch_size 4 \
--learning_rate 1e-6 \
--max_length 128 > output.log 2>&1 &
```
- train lora:

```
nohup python train_lora.py --data_file data/sarcasm.csv \
--output_dir logs \
--num_train_epochs 2 \
--batch_size 2 \
--learning_rate 1e-5 \
--max_length 128 \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 > output.log 2>&1 &
```
- view log: `tail -f output.log`

- view `tensorboard`: `tensorboard --logdir logs/logs`
- test model: `python test.py --data_file_path ./data/sarcasm.csv --model_id fine-tuned-model --cuda_device 0 --max_length 128`

- inference model: `python inference.py logs/checkpoint-190`

before finetune

```
{'BERTScore': {'Precision': 0.8277986645698547, 'Recall': 0.8530203700065613, 'F1': 0.8401092886924744}, 'ROUGE-L': {'F1': 0.0794529113715608, 'Precision': 0.057397086892784416, 'Recall': 0.16667777018694505}, 'BLEU-4': 0.0027716679935327476, 'F1-Score': 0.8402202725410461}
```

after finetune

```
{'BERTScore': {'Precision': 0.8509647250175476, 'Recall': 0.876897931098938, 'F1': 0.863616406917572}, 'ROUGE-L': {'F1': 0.1416847936719217, 'Precision': 0.10668925006579812, 'Recall': 0.23828849169350153}, 'BLEU-4': 0.22531999050937368, 'F1-Score': 0.8637367486953735}
```

*Note: RuntimeError: TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX.*

# references

[llama-fine-tune-guide](https://github.com/AlexandrosChrtn/llama-fine-tune-guide/tree/main)

[prompt-engineering-with-llama-2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/)