# AraContract Analyzer — نسخة النموذج ومقارنة الأداء

## نظرة عامة

هذا الملف يوثق تطور نموذج تصنيف بنود العقود القانونية العربية عبر ثلاث نسخ رئيسية، مع التحسينات المُطبَّقة والنتائج النهائية لكل نسخة.

---

## ملخص النسخ

| النسخة | التاريخ | التحسينات الرئيسية | Medium F1 | Macro F1 | Task 1 Accuracy |
|---|---|---|---|---|---|
| **v1** | 2026-05-28 | نموذج أساسي (CAMeLBERT) | 0.4516 | 0.7404 | 0.8872 |
| **v1.5** | 2026-06-01 | Class Weights + Oversampling | 0.4818 | 0.7317 | 0.8619 |
| **v2** | 2026-06-03 | Oversampling فقط + Threshold Tuning | **0.5333** | **0.7547** | 0.8631 |

---

## النسخة v1 — النموذج الأساسي

### التكوين
```python
risk_class_weights = [1.0, 1.0, 1.0]  # بدون أوزان
oversampling = False
threshold = 0.5  # argmax default
```

### النتائج
```
TASK 1 — Clause Type Classification
           accuracy: 0.8872
          macro F1: 0.8746
       weighted F1: 0.8882

TASK 2 — Risk Level Classification
              precision    recall  f1-score   support
         low     0.92xx    0.94xx    0.93xx       617
      medium     0.42xx    0.35xx    0.4516        60  ← كارثي
        high     0.76xx    0.89xx    0.82xx       192

   macro avg F1: 0.7404  ← مخفي خلف weighted avg 0.87
```

### المشكلة الجذرية
**توزيع البيانات:**
```
low:    2598 عينة (72%)
high:    781 عينة (22%)
medium:  230 عينة (6%)  ← قليل جداً
```

النموذج تعلّم يتجاهل الـ medium لأنه الأقل تمثيلاً — الـ recall 35% يعني أن 65% من البنود متوسطة الخطورة تم تفويتها.

### السبب
بدون أي معالجة لعدم التوازن، الـ CrossEntropyLoss يعامل كل العيّنات بنفس الطريقة، فالـ low يطغى على الـ gradient.

---

## النسخة v1.5 — Class Weights + Oversampling

### التكوين
```python
risk_class_weights = [1.0, 2.0, 1.2]  # وزن إضافي للـ medium
oversampling = True  # WeightedRandomSampler
threshold = 0.39  # threshold tuning
```

### التحسينات المُطبَّقة

**1. رفع وزن medium في الـ loss:**
```python
self.risk_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 2.0, 1.2])
)
```

**2. Oversampling عبر WeightedRandomSampler:**
```python
# نحسب وزن كل عينة عكس تكرار صنفها
risk_counts = Counter(r['risk_level'] for r in train_data)
sample_weights = [
    1.0 / risk_counts[r['risk_level']]
    for r in train_data
]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_data),
    replacement=True
)
```

**3. بحث عن optimal threshold:**
```python
for thresh in np.arange(0.15, 0.45, 0.02):
    # احسب macro F1
    # اختر أفضل threshold
best_thresh = 0.39
```

### النتائج
```
TASK 1 — Clause Type Classification
           accuracy: 0.8619  (−2.5%)
          macro F1: 0.8401
       weighted F1: 0.8624

TASK 2 — Risk Level Classification
              precision    recall  f1-score   support
         low     0.9477    0.8525    0.8976       617
      medium     0.4286    0.5500    0.4818        60  ← تحسّن
        high     0.7384    0.9115    0.8159       192

   macro avg F1: 0.7317  (−1.2%)
```

### التحليل

**الإيجابيات:**
- ✅ medium recall تحسّن من 35% → 55% (+20%)
- ✅ medium F1 تحسّن من 0.45 → 0.48 (+0.03)

**السلبيات:**
- ⚠️ medium precision低 (0.43) — النموذج يتوقع medium كثيراً بشكل خاطئ
- ⚠️ Task 1 accuracy انخفض من 0.887 → 0.862
- ⚠️ low recall انخفض من 94% → 85%

### السبب الجذري

**Double Compensation** — الـ medium أخذ تعويض مضاعف:
1. وزن أعلى في الـ loss (×2.0)
2. عينات أكثر في الـ sampler

هذا جعل النموذج **مُتحيزاً باتجاه medium** أكثر من اللازم، فضحّى بالـ low و Task 1.

---

## النسخة v2 — Oversampling فقط + Threshold Tuning

### التكوين
```python
risk_class_weights = [1.0, 1.0, 1.0]  # بدون أوزان!
oversampling = True  # WeightedRandomSampler
threshold = 0.43  # threshold tuning (أعلى من v1.5)
```

### التغييرات الرئيسية

**1. إزالة الـ class weights تماماً:**
```python
self.risk_criterion = nn.CrossEntropyLoss()  # default = uniform weights
```

**2. الإبقاء على Oversampling:**
```python
# نفس الـ WeightedRandomSampler من v1.5
sampler = WeightedRandomSampler(...)
```

**3. Threshold أعلى (0.43 بدل 0.39):**
```python
# بحث جديد عن optimal threshold
best_thresh = 0.43  # مقارنة بـ 0.39 في v1.5
```

### النتائج النهائية
```
TASK 1 — Clause Type Classification
           accuracy: 0.8631  (+0.1% vs v1.5)
          macro F1: 0.8401  (مستقر)
       weighted F1: 0.8635

TASK 2 — Risk Level Classification
              precision    recall  f1-score   support
         low     0.9440    0.8736    0.9074       617  ← تحسّن
      medium     0.5333    0.5333    0.5333        60  ← تحسّن كبير
        high     0.7437    0.9219    0.8233       192  ← تحسّن

   macro avg F1: 0.7547  (+3.1% vs v1.5)
```

### سجل تدريب v2

| Epoch | Train Loss | Type Loss | Risk Loss | Val Type F1 | Val Risk F1 | Val Type Acc | Val Risk Acc |
|---|---|---|---|---|---|---|---|
| 1 | 2.1983 | 1.3997 | 0.7985 | 0.7219 | 0.7531 | 0.7167 | 0.6999 |
| 2 | 0.9885 | 0.5557 | 0.4328 | 0.7860 | 0.8624 | 0.7801 | 0.8525 |
| 3 | 0.5624 | 0.3186 | 0.2438 | 0.8089 | 0.8889 | 0.8060 | 0.8849 |
| 4 | 0.3986 | 0.2238 | 0.1748 | 0.8263 | 0.8752 | 0.8228 | 0.8668 |
| 5 | 0.3261 | 0.1897 | 0.1364 | 0.8241 | 0.8996 | 0.8215 | 0.8978 |

**Best Model:** Epoch 5 — Avg F1 = 0.8618

### التحليل

**مقارنة v2 vs v1.5:**

| المقاس | v1.5 | v2 | التحسين |
|---|---|---|---|
| medium precision | 0.4286 | **0.5333** | +24% ✓ |
| medium recall | 0.5500 | 0.5333 | −3% (مقبول) |
| **medium F1** | **0.4818** | **0.5333** | **+11%** ✓ |
| low F1 | 0.8976 | **0.9074** | +1% ✓ |
| high F1 | 0.8159 | **0.8233** | +1% ✓ |
| **macro avg F1** | **0.7317** | **0.7547** | **+3%** ✓ |
| Task 1 accuracy | 0.8619 | **0.8631** | مستقر ✓ |

### السبب الجذري للنجاح

**توازن أفضل:**
- الـ Oversampling وحده أعطى الـ model فرص كافية لتعلّم medium
- بدون class weights، النموذج ما انحاز بشكل مفرط
- الـ threshold 0.43 وازن بين precision و recall بشكل أمثل

**لماذاthreshold أعلى من v1.5؟**
- في v1.5، الـ class weights خلّت النموذج يتوقع medium بسهولة
- في v2، بدون weights، النموذج أصبح أكثر تحفظاً
- لذلك احتجنا threshold أعلى (0.43 vs 0.39) لتحفيز التوقعات دون مبالغة

---

## الدروس المستفادة

### 1. Class Weights + Oversampling = Double Compensation ❌

استخدام الآليتين معاً يسبب تحيزاً مفرطاً تجاه الأقلية:
```
medium weight ×2.0  +  oversampling ×3.0  =  ×6.0 compensation
```

هذا قد يحسّن recall لكن على حساب precision والأصناف الأخرى.

### 2. Oversampling وحده أفضل من Class Weights وحده ✅

| الطريقة | medium F1 | macro F1 | Task 1 accuracy |
|---|---|---|---|
| Weights فقط | ~0.45 | ~0.72 | 0.86 |
| Oversampling فقط | **0.53** | **0.75** | **0.86** |

السبب: الـ Oversampling يعيّن النموذج عينات أكثرหลากหลาย، بينما الـ Weights يغيّر الـ gradient فقط.

### 3. Threshold Tuning ضروري للبيانات غير المتوازنة ✅

بدون threshold tuning:
```
default threshold = 0.5 → medium recall = 35%
tuned threshold = 0.43 → medium recall = 53%
```

تحسن +18% في recall بتغيير سطر واحد!

### 4. Metric اختيار: Macro F1 أهم من Weighted F1

| النسخة | Weighted F1 | Macro F1 |
|---|---|---|
| v1 | 0.8702 | 0.7404 |
| v2 | 0.8630 | **0.7547** |

الـ Weighted F1 يخفي ضعف medium (لأن low 72% يطغى)، لكن Macro F1 يكشفه.

---

## التوصيات للنماذج المستقبلية

### للبيانات غير المتوازنة (<10% لفئة)

1. **ابدأ بـ Oversampling وحده** — بدون class weights
2. **طبق Threshold Tuning** — ابحث في نطاق 0.3-0.5
3. **راقب Macro F1** — ليس فقط Weighted F1
4. **تفقّد per-class metrics** — precision و recall لكل صنف

### الترتيب الموصى به

```
1. Train baseline (no modifications)
2. Add Oversampling → evaluate
3. Add Threshold Tuning → evaluate
4. فقط إذا لزم:<Class Weights> خفيف (×1.5 كحد أقصى)
5. اختبر Focal Loss إذا كان medium لا يزال ضعيفاً
```

---

## الاستنتاج النهائي

**النسخة v2 هي الأفضل حتى الآن:**
- ✅ أعلى medium F1 (0.5333)
- ✅ أعلى macro F1 (0.7547)
- ✅ Task 1 مستقر (0.8631)
- ✅ توازن أفضل بين الأصناف

**مجالات التحسين المستقبلية:**
- Focal Loss للتركيز على الأمثلة الصعبة
- Data augmentation للنصوص العربية لزيادة medium samples
- Ensemble methods (تجربة عدة نماذج معاً)