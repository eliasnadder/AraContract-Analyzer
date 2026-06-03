 No checkpoint found. Starting training from scratch.
Using device: cuda
Loading tokenizer: CAMeL-Lab/bert-base-arabic-camelbert-msa
/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
tokenizer_config.json: 100% 86.0/86.0 [00:00<00:00, 6.97kB/s]config.json: 100% 468/468 [00:00<00:00, 26.1kB/s]vocab.txt:  305k/? [00:00<00:00, 11.6MB/s]special_tokens_map.json: 100% 112/112 [00:00<00:00, 12.8kB/s]Building dataloaders...
Risk counts for sampler: {'high': 781, 'low': 2598, 'medium': 230}
Sample weight stats: min=0.46, max=5.23
Train batches: 226, Val batches: 49
Loading model: CAMeL-Lab/bert-base-arabic-camelbert-msa
/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
pytorch_model.bin: 100% 439M/439M [00:08<00:00, 67.8MB/s]/tmp/ipykernel_8812/738197880.py:80: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
/tmp/ipykernel_8812/1651819197.py:102: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():

Starting training for 5 epochs...
Run name: aracontract_v2
Batch size: 16, LR: 2e-05

==================================================
Epoch 1/5
==================================================
/tmp/ipykernel_8812/1651819197.py:115: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  scheduler.step()
  Step 50/226 | Loss: 3.0125
  Step 100/226 | Loss: 2.7141
  Step 150/226 | Loss: 2.5031
  Step 200/226 | Loss: 2.2979
Train loss: 2.1983 | type: 1.3997 | risk: 0.7985
Val type_f1: 0.7219 | risk_f1: 0.7531
Val type_acc: 0.7167 | risk_acc: 0.6999
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
  ★ New best model! Avg F1: 0.7375
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_epoch1.pt

==================================================
Epoch 2/5
==================================================
/tmp/ipykernel_8812/1651819197.py:102: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
  Step 50/226 | Loss: 1.2363
  Step 100/226 | Loss: 1.1382
  Step 150/226 | Loss: 1.0587
  Step 200/226 | Loss: 1.0133
Train loss: 0.9885 | type: 0.5557 | risk: 0.4328
Val type_f1: 0.7860 | risk_f1: 0.8624
Val type_acc: 0.7801 | risk_acc: 0.8525
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
  ★ New best model! Avg F1: 0.8242
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_epoch2.pt

==================================================
Epoch 3/5
==================================================
/tmp/ipykernel_8812/1651819197.py:102: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
  Step 50/226 | Loss: 0.6752
  Step 100/226 | Loss: 0.6244
  Step 150/226 | Loss: 0.5843
  Step 200/226 | Loss: 0.5778
Train loss: 0.5624 | type: 0.3186 | risk: 0.2438
Val type_f1: 0.8089 | risk_f1: 0.8889
Val type_acc: 0.8060 | risk_acc: 0.8849
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
  ★ New best model! Avg F1: 0.8489
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_epoch3.pt

==================================================
Epoch 4/5
==================================================
/tmp/ipykernel_8812/1651819197.py:102: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
  Step 50/226 | Loss: 0.4602
  Step 100/226 | Loss: 0.4409
  Step 150/226 | Loss: 0.4285
  Step 200/226 | Loss: 0.4089
Train loss: 0.3986 | type: 0.2238 | risk: 0.1748
Val type_f1: 0.8263 | risk_f1: 0.8752
Val type_acc: 0.8228 | risk_acc: 0.8668
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
  ★ New best model! Avg F1: 0.8507
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_epoch4.pt

==================================================
Epoch 5/5
==================================================
/tmp/ipykernel_8812/1651819197.py:102: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
  Step 50/226 | Loss: 0.3581
  Step 100/226 | Loss: 0.3544
  Step 150/226 | Loss: 0.3368
  Step 200/226 | Loss: 0.3302
Train loss: 0.3261 | type: 0.1897 | risk: 0.1364
Val type_f1: 0.8241 | risk_f1: 0.8996
Val type_acc: 0.8215 | risk_acc: 0.8978
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
  ★ New best model! Avg F1: 0.8618
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_epoch5.pt
  Checkpoint saved to /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_final.pt

==================================================
Training complete!
Best weighted F1: 0.8618
Checkpoints saved to: /content/drive/MyDrive/AraContract/checkpoints
==================================================

Loaded checkpoint: /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
Run name: aracontract_v2
/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
✓ Model loaded

Evaluating on test set...
Test type_f1: 0.8635 | risk_f1: 0.8658
Test type_acc: 0.8631 | risk_acc: 0.8642
✓ Meets SRS target


Loaded checkpoint for evaluation: /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Best medium threshold: 0.43, Macro F1: 0.7547
=======================================================
TASK 1 — Clause Type Classification
=======================================================
                     precision    recall  f1-score   support

 general_provisions     0.8714    0.9385    0.9037       130
  payment_financial     0.9021    0.8663    0.8838       202
party_obligations_a     0.6462    0.6774    0.6614        62
party_obligations_b     0.6905    0.7838    0.7342        37
duration_expiration     0.8629    0.8231    0.8425       130
        termination     0.8495    0.8316    0.8404        95
  penalties_damages     0.9350    0.8915    0.9127       129
 dispute_resolution     0.9205    0.9643    0.9419        84

           accuracy                         0.8631       869
          macro avg     0.8347    0.8471    0.8401       869
       weighted avg     0.8653    0.8631    0.8635       869

=======================================================
TASK 2 — Risk Level Classification (threshold-adjusted)
=======================================================
              precision    recall  f1-score   support

         low     0.9440    0.8736    0.9074       617
      medium     0.5333    0.5333    0.5333        60
        high     0.7437    0.9219    0.8233       192

    accuracy                         0.8608       869
   macro avg     0.7403    0.7763    0.7547       869
weighted avg     0.8714    0.8608    0.8630       869


Clause: يلتزم الطرف الثاني بدفع المبلغ المتفق عليه خلال مد...
  Type: payment_financial (76.33%)
  Risk: low (92.66%)

Clause: يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مس...
  Type: general_provisions (23.66%)
  Risk: low (88.02%)

Clause: تبلغ مدة هذا العقد سنة واحدة قابلة للتجديد تلقائيا...
  Type: duration_expiration (67.03%)
  Risk: low (97.59%)
