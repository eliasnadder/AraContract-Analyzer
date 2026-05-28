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
Checkpoints saved to: /content/drive/MyDrive/AraContract/checkpoints
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
