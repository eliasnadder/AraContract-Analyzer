# Training

Starting training for 5 epochs...
Run name: aracontract_v1
Batch size: 16, LR: 2e-05

==================================================
Epoch 1/5
==================================================
  Step 50/249 | Loss: 2.7268
  Step 100/249 | Loss: 2.4915
  Step 150/249 | Loss: 2.3003
  Step 200/249 | Loss: 2.1163
Train loss: 1.9733 | type: 1.3500 | risk: 0.6233
Val type_f1: 0.7095 | risk_f1: 0.7974
Val type_acc: 0.7174 | risk_acc: 0.8039

==================================================
Epoch 2/5
==================================================
  Step 50/249 | Loss: 1.2232
  Step 100/249 | Loss: 1.1314
  Step 150/249 | Loss: 1.0754
  Step 200/249 | Loss: 1.0430
Train loss: 1.0252 | type: 0.6079 | risk: 0.4173
Val type_f1: 0.8708 | risk_f1: 0.8184
Val type_acc: 0.8697 | risk_acc: 0.8307

==================================================
Epoch 3/5
==================================================
  Step 50/249 | Loss: 0.7660
  Step 100/249 | Loss: 0.7552
  Step 150/249 | Loss: 0.7273
  Step 200/249 | Loss: 0.7116
Train loss: 0.7017 | type: 0.3684 | risk: 0.3332
Val type_f1: 0.8668 | risk_f1: 0.8438
Val type_acc: 0.8685 | risk_acc: 0.8563

==================================================
Epoch 4/5
==================================================
  Step 50/249 | Loss: 0.5850
  Step 100/249 | Loss: 0.5670
  Step 150/249 | Loss: 0.5677
  Step 200/249 | Loss: 0.5528
Train loss: 0.5434 | type: 0.2657 | risk: 0.2777
Val type_f1: 0.9016 | risk_f1: 0.8599
Val type_acc: 0.9013 | risk_acc: 0.8660

==================================================
Epoch 5/5
==================================================
  Step 50/249 | Loss: 0.4530
  Step 100/249 | Loss: 0.4319
  Step 150/249 | Loss: 0.4410
  Step 200/249 | Loss: 0.4337
Train loss: 0.4260 | type: 0.1935 | risk: 0.2325
Val type_f1: 0.9064 | risk_f1: 0.8519
Val type_acc: 0.9062 | risk_acc: 0.8575

==================================================
Training complete!
Best weighted F1: 0.8807
==================================================

✓ Model loaded
Evaluating on test set...
Test type_f1: 0.8699 | risk_f1: 0.8485
Test type_acc: 0.8701 | risk_acc: 0.8586
✓ Meets SRS target

=== Type Clause Classification Report ===
                     precision    recall  f1-score   support

 general_provisions     0.9638    0.9568    0.9603       139
  payment_financial     0.8795    0.8639    0.8716       169
  party_obligations     0.7904    0.9471    0.8617       227
duration_expiration     0.8516    0.8583    0.8549       127
        termination     0.7429    0.6667    0.7027        78
  penalties_damages     0.9652    0.8043    0.8775       138
 dispute_resolution     0.9726    0.8452    0.9045        84

           accuracy                         0.8701       962
          macro avg     0.8809    0.8489    0.8619       962
       weighted avg     0.8763    0.8701    0.8699       962

Clause: يلتزم الطرف الثاني بدفع المبلغ المتفق عليه خلال مد...
  Type: payment_financial (53.82%)
  Risk: low (62.37%)

Clause: يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مس...
  Type: duration_expiration (43.64%)
  Risk: low (82.16%)

Clause: تبلغ مدة هذا العقد سنة واحدة قابلة للتجديد تلقائيا...
  Type: duration_expiration (78.96%)
  Risk: low (93.39%)

---

## After split party obligations
```text
Starting training for 5 epochs...
Run name: aracontract_v1
Batch size: 16, LR: 2e-05

==================================================
Epoch 1/5
==================================================
  Step 50/226 | Loss: 2.8229
  Step 100/226 | Loss: 2.5011
  Step 150/226 | Loss: 2.2584
  Step 200/226 | Loss: 2.0642
Train loss: 1.9824 | type: 1.3751 | risk: 0.6072
Val type_f1: 0.7179 | risk_f1: 0.8394
Val type_acc: 0.7257 | risk_acc: 0.8577
  ★ New best model! Avg F1: 0.7787
```
```text
==================================================
Epoch 2/5
==================================================
  Step 50/226 | Loss: 1.1890
  Step 100/226 | Loss: 1.1208
  Step 150/226 | Loss: 1.0836
  Step 200/226 | Loss: 1.0563
Train loss: 1.0379 | type: 0.6632 | risk: 0.3747
Val type_f1: 0.8075 | risk_f1: 0.8765
Val type_acc: 0.8060 | risk_acc: 0.8862
  ★ New best model! Avg F1: 0.8420
```
```text
==================================================
Epoch 3/5
==================================================
  Step 50/226 | Loss: 0.7171
  Step 100/226 | Loss: 0.7270
  Step 150/226 | Loss: 0.7215
  Step 200/226 | Loss: 0.7010
Train loss: 0.6993 | type: 0.4210 | risk: 0.2783
Val type_f1: 0.8315 | risk_f1: 0.8921
Val type_acc: 0.8305 | risk_acc: 0.8978
  ★ New best model! Avg F1: 0.8618
```
```text
==================================================
Epoch 4/5
==================================================
  Step 50/226 | Loss: 0.4991
  Step 100/226 | Loss: 0.5208
  Step 150/226 | Loss: 0.5218
  Step 200/226 | Loss: 0.5088
Train loss: 0.5135 | type: 0.2888 | risk: 0.2247
Val type_f1: 0.8593 | risk_f1: 0.9022
Val type_acc: 0.8577 | risk_acc: 0.9069
  ★ New best model! Avg F1: 0.8807
```
```text
==================================================
Epoch 5/5
==================================================
  Step 50/226 | Loss: 0.4198
  Step 100/226 | Loss: 0.4198
  Step 150/226 | Loss: 0.4102
  Step 200/226 | Loss: 0.4118
Train loss: 0.4123 | type: 0.2259 | risk: 0.1864
Val type_f1: 0.8604 | risk_f1: 0.9046
Val type_acc: 0.8590 | risk_acc: 0.9082
  ★ New best model! Avg F1: 0.8825

==================================================
Training complete!
Best weighted F1: 0.8825
==================================================
```

Loaded checkpoint: /content/drive/MyDrive/AraContract/checkpoints/aracontract_v1_best.pt
Run name: aracontract_v1

Evaluating on test set...
Test type_f1: 0.8882 | risk_f1: 0.8702
Test type_acc: 0.8872 | risk_acc: 0.8757
✓ Meets SRS target

```text
=== Type Clause Classification Report ===
                     precision    recall  f1-score   support

 general_provisions     0.9308    0.9308    0.9308       130
  payment_financial     0.9000    0.8911    0.8955       202
party_obligations_a     0.6575    0.7742    0.7111        62
party_obligations_b     0.8649    0.8649    0.8649        37
duration_expiration     0.8702    0.8769    0.8736       130
        termination     0.8736    0.8000    0.8352        95
  penalties_damages     0.9444    0.9225    0.9333       129
 dispute_resolution     0.9529    0.9643    0.9586        84

           accuracy                         0.8872       869
          macro avg     0.8743    0.8781    0.8754       869
       weighted avg     0.8902    0.8872    0.8882       869
```
### Test
Clause: يلتزم الطرف الثاني بدفع المبلغ المتفق عليه خلال مد...
  Type: payment_financial (83.12%)
  Risk: low (95.15%)

Clause: يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مس...
  Type: party_obligations_b (25.49%)
  Risk: low (88.23%)

Clause: تبلغ مدة هذا العقد سنة واحدة قابلة للتجديد تلقائيا...
  Type: duration_expiration (83.68%)
  Risk: low (97.23%)

---

## Third training
Using device: cuda
Loading tokenizer: CAMeL-Lab/bert-base-arabic-camelbert-msa
Building dataloaders...
Train batches: 226, Val batches: 49

Starting training for 5 epochs...
Run name: aracontract_v2
Batch size: 16, LR: 2e-05

==================================================
Epoch 1/5
==================================================
  Step 50/226 | Loss: 2.9940
  Step 100/226 | Loss: 2.7236
  Step 150/226 | Loss: 2.4958
  Step 200/226 | Loss: 2.2981
Train loss: 2.2210 | type: 1.4531 | risk: 0.7679
Val type_f1: 0.7220 | risk_f1: 0.7885
Val type_acc: 0.7219 | risk_acc: 0.7542
  ★ New best model! Avg F1: 0.7553

==================================================
Epoch 2/5
==================================================
  Step 50/226 | Loss: 1.4119
  Step 100/226 | Loss: 1.3211
  Step 150/226 | Loss: 1.2694
  Step 200/226 | Loss: 1.2264
Train loss: 1.2054 | type: 0.7255 | risk: 0.4800
Val type_f1: 0.8002 | risk_f1: 0.8745
Val type_acc: 0.7969 | risk_acc: 0.8758
  ★ New best model! Avg F1: 0.8374

==================================================
Epoch 3/5
==================================================
  Step 50/226 | Loss: 0.8347
  Step 100/226 | Loss: 0.8492
  Step 150/226 | Loss: 0.8525
  Step 200/226 | Loss: 0.8297
Train loss: 0.8294 | type: 0.4768 | risk: 0.3526
Val type_f1: 0.8230 | risk_f1: 0.8834
Val type_acc: 0.8189 | risk_acc: 0.8797
  ★ New best model! Avg F1: 0.8532

==================================================
Epoch 4/5
==================================================
  Step 50/226 | Loss: 0.6083
  Step 100/226 | Loss: 0.6281
  Step 150/226 | Loss: 0.6368
  Step 200/226 | Loss: 0.6154
Train loss: 0.6219 | type: 0.3467 | risk: 0.2752
Val type_f1: 0.8556 | risk_f1: 0.9016
Val type_acc: 0.8525 | risk_acc: 0.8991
  ★ New best model! Avg F1: 0.8786

==================================================
Epoch 5/5
==================================================
  Step 50/226 | Loss: 0.5186
  Step 100/226 | Loss: 0.5035
  Step 150/226 | Loss: 0.4951
  Step 200/226 | Loss: 0.4962
Train loss: 0.4955 | type: 0.2793 | risk: 0.2162
Val type_f1: 0.8551 | risk_f1: 0.9059
Val type_acc: 0.8525 | risk_acc: 0.9030
  ★ New best model! Avg F1: 0.8805

==================================================
Training complete!
Best weighted F1: 0.8805
==================================================

Loaded checkpoint: /content/drive/MyDrive/AraContract/checkpoints/aracontract_v2_best.pt
Run name: aracontract_v2

Evaluating on test set...
Test type_f1: 0.8732 | risk_f1: 0.8617
Test type_acc: 0.8711 | risk_acc: 0.8585
✓ Meets SRS target

=== Type Clause Classification Report ===
                     precision    recall  f1-score   support

 general_provisions     0.9440    0.9077    0.9255       130
  payment_financial     0.9149    0.8515    0.8821       202
party_obligations_a     0.6265    0.8387    0.7172        62
party_obligations_b     0.7778    0.7568    0.7671        37
duration_expiration     0.8672    0.8538    0.8605       130
        termination     0.8061    0.8316    0.8187        95
  penalties_damages     0.9280    0.8992    0.9134       129
 dispute_resolution     0.9419    0.9643    0.9529        84

           accuracy                         0.8711       869
          macro avg     0.8508    0.8629    0.8547       869
       weighted avg     0.8784    0.8711    0.8732       869

Clause: يلتزم الطرف الثاني بدفع المبلغ المتفق عليه خلال مد...
  Type: payment_financial (73.68%)
  Risk: low (93.44%)

Clause: يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مس...
  Type: party_obligations_b (21.44%)
  Risk: low (88.66%)

Clause: تبلغ مدة هذا العقد سنة واحدة قابلة للتجديد تلقائيا...
  Type: duration_expiration (70.39%)
  Risk: low (96.00%)