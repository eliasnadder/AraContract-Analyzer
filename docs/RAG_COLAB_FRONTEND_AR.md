# دليل تشغيل RAG للعقود العربية على Google Colab

يوضح هذا الدليل كيفية تشغيل نظام الإجابة عن أسئلة العقود (RAG) في Google Colab، ثم ربطه بواجهة المستخدم العربية.

## الملفات المطلوبة

| الملف | الغرض |
|---|---|
| `backend/AraContract_RAG_Frontend_Colab.ipynb` | دفتر Colab لتشغيل Ollama وRAG وngrok والواجهة التجريبية. |
| `backend/colab_rag_server.py` | خادم FastAPI خفيف لمسارات RAG فقط. |
| `frontend/rag-client.html` | واجهة عربية RTL جاهزة للاختبار أو الدمج. |

## المتطلبات

1. حساب Google لاستخدام Colab.
2. حساب ngrok مجاني للحصول على Auth Token من <https://dashboard.ngrok.com/get-started/your-authtoken>.
3. مستودع المشروع على GitHub، أو نسخة منه مرفوعة/مربوطة بـ Google Drive.
4. يوصى بتفعيل GPU من Colab:

   `Runtime → Change runtime type → T4 GPU`

> جلسة Colab ورابط ngrok مؤقتان. لا ترفع عقوداً حساسة إلى رابط عام أو تشاركه مع غير المصرح لهم.

## التشغيل على Colab

1. ارفع أو افتح الملف `backend/AraContract_RAG_Frontend_Colab.ipynb` في Google Colab.
2. شغّل الخلايا بالترتيب من الأعلى إلى الأسفل.
3. في خلية تثبيت التبعيات، سيجري تنزيل مكتبات FastAPI وQdrant وSentence Transformers وغيرها.
4. في خلية تنزيل النماذج، سيجري حفظ نماذج الـ embeddings وإعادة الترتيب داخل:

   ```text
   backend/models_local/
   ```

5. في خلية Ollama، سيتم تثبيت Ollama وتشغيله ثم تنزيل نموذج `qwen2.5:3b`.
6. عند ظهور مطالبة `ngrok auth token:`، ألصق الرمز الذي نسخته من لوحة تحكم ngrok واضغط Enter.
7. ستطبع الخلية رابطاً شبيهاً بالآتي:

   ```text
   RAG API: https://xxxx.ngrok-free.app/api/contract/rag
   ```

احتفظ بهذا الرابط؛ فهو قيمة `API_BASE` التي ستستخدمها الواجهة الأمامية.

## سير عمل RAG

```text
بنود العقد → /ingest → session_id → /ask → الإجابة والمصادر
                                      ↓
                              DELETE /session/{session_id}
```

### 1. استيعاب بنود العقد

أرسل البنود، بنداً مستقلاً في كل عنصر من المصفوفة. يمكن الحصول عليها مسبقاً من مسار التقسيم في النظام أو من نص العقد مباشرة.

```http
POST {API_BASE}/ingest
Content-Type: application/json
```

```json
{
  "clauses": [
    "المادة الأولى: يلتزم الطرف الأول بتقديم الخدمة خلال خمسة أيام عمل.",
    "المادة الثانية: يسدد الطرف الثاني المستحقات خلال ثلاثين يوماً."
  ]
}
```

مثال للاستجابة:

```json
{
  "session_id": "c2d6d6f5-0a73-4ee3-bf9d-7b2a2b9a1e77",
  "clauses_count": 2,
  "message": "تم استيعاب العقد بنجاح وبناء الـ Vector Store"
}
```

احفظ `session_id` في حالة الواجهة (state) لأنه مطلوب لكل سؤال لاحق.

### 2. طرح سؤال عن العقد

```http
POST {API_BASE}/ask
Content-Type: application/json
```

```json
{
  "session_id": "c2d6d6f5-0a73-4ee3-bf9d-7b2a2b9a1e77",
  "question": "ما هي مهلة السداد؟",
  "top_k": 3
}
```

تتضمن الاستجابة `answer` و`retrieved_clauses`. اعرض المصادر المسترجعة دائماً للمستخدم حتى يستطيع التحقق من الإجابة في نص العقد.

### 3. حذف الجلسة

بعد انتهاء المستخدم، احذف مخزن المتجهات المؤقت لتوفير ذاكرة Colab:

```http
DELETE {API_BASE}/session/{session_id}
```

## تجربة الواجهة الجاهزة

يعرض الدفتر نفسه واجهة `frontend/rag-client.html` بعد استبدال `window.__RAG_API_BASE__` برابط ngrok الناتج. لا تحتاج إلى إعداد إضافي لتجربتها من داخل Colab.

تتيح الواجهة:

- إدخال بند واحد في كل سطر.
- إنشاء جلسة RAG.
- طرح الأسئلة بالعربية.
- عرض الإجابة والبنود المسترجعة مع درجة الصلة.
- إنهاء الجلسة.

## دمج الواجهة في تطبيقك

انسخ محتوى `frontend/rag-client.html` أو أضفه كصفحة مستقلة. قبل تحميل منطق الصفحة، اضبط الرابط العام:

```html
<script>
  window.__RAG_API_BASE__ = "https://xxxx.ngrok-free.app/api/contract/rag";
</script>
```

إذا كان التطبيق مبنياً بـ React أو Vue أو Flutter Web، اتبع نفس التسلسل: استدعِ `/ingest` مرة واحدة، احفظ `session_id`، ثم أرسله مع كل طلب إلى `/ask`، ونظّف الجلسة عند مغادرة صفحة المحادثة.

## فحص الحالة وحل المشاكل

تحقق من أن الخادم ونموذج اللغة جاهزان:

```text
GET {API_BASE}/status
```

| المشكلة | الحل |
|---|---|
| `ready: false` | أعد تشغيل خلية Ollama، ثم تأكد أن تنزيل النموذج اكتمل. |
| خطأ في ngrok | تأكد من صحة Auth Token وأعد تشغيل خلية النفق. |
| انتهى رابط ngrok | أعد تشغيل خلية ngrok وحدّث `API_BASE` في الواجهة. |
| نفاد ذاكرة GPU | استخدم نموذج Ollama أصغر أو أعد تشغيل Runtime. |
| `Session not found` | أنشئ جلسة جديدة عبر `/ingest` واحفظ `session_id` الجديد. |
| لا تظهر الواجهة | شغّل خلية الواجهة بعد خلية ngrok وتأكد أن `API_BASE` غير فارغ. |

## ملاحظات أمنية

- مسارات RAG الحالية عامة؛ رابط ngrok ليس بديلاً عن المصادقة.
- استخدم رابطاً خاصاً أو أضف مصادقة قبل استخدام الحل مع بيانات إنتاجية.
- لا تحفظ `session_id` كبيانات دائمة؛ هو صالح فقط ما دامت جلسة الخادم موجودة.
- احذف الجلسة عند نهاية المحادثة، وأوقف Colab/ngrok عند الانتهاء.
