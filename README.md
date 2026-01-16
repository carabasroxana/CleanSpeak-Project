Task 7: Everyday Knowledge Across Diverse Languages and Cultures
Track 2: Multiple-Choice Questions (MCQ)

Roxana Carabas & Anamaria Nacu



1. Problem Description

SemEval-2026 Task 7 focuses on evaluating how well language models know everyday cultural knowledge across many languages and regions. Existing LLMs are powerful, but they often reflect Western-centric information and fail on simple questions about food, holidays or daily routines in under-represented cultures. 

The shared task is based on an extended version of the BLEnD benchmark (“Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages”). BLEnD is a hand-crafted dataset of about 52.6k question-answer pairs covering 16 countries/regions and 13 languages, including low-resource ones like Amharic, Assamese, Azerbaijani and Hausa. Questions are about common-sense cultural knowledge (typical birthday food, greeting customs, school sports, etc.) and are provided both as short-answer questions and as multiple-choice questions (MCQ). 

SemEval-2026 Task 7 extends BLEnD to more than 30 language–culture pairs by adding many new regions while keeping BLEnD itself for dev/test only, so that systems are evaluated on unseen questions. 

In Track 2 (MCQ):

Input: Each instance is an English question about everyday life plus four answer options. 

Cultural framing: Each option reflects how one specific region would answer that question. The lang_reg code (e.g. el-GR, es-MX) says which region we care about on this instance. 

Output / label: The model must select the option that matches the everyday knowledge of that region; accuracy over all questions is the official metric. 

The pilot data for Track 2 is released as trial_data_multiple_choice.tsv with the following columns: index, lang_reg, question, multiple_choice_options (newline-separated options) and correct_answer. There are 23 possible language–region codes listed in the README (e.g. en-GB, es-EC, ar-SA, ja-JP, etc.). 


2. State of the Art Review

2.1.Existing datasets and benchmarks

Besides BLEnD / SemEval Task 7, there is a small but fast-growing ecosystem of cultural benchmarks:
 - BLEnD (base dataset for this task).
 - 52.6k QA pairs, 16 regions, 13 languages.
- Two formats: short-answer and MCQ.


Evaluation of 16 LLMs shows:
- Huge variation across cultures: for GPT-4, the gap between the best and worst cultures in the short-answer task is 57.34 percentage points.
- LLMs perform better in their local language for mid/high-resource cultures, but for low-resource languages they often do better in English, reflecting the training data imbalance. 

CulturalBench – a diverse MCQ benchmark.
- 1,227 human-written & human-verified questions about culture across 45 world regions and 17 topics (food, etiquette, religion, etc.).
- Two “phrasings”: Easy and Hard, which use the same content but different question forms.
- Humans reach ~92.6% accuracy; on the harder version the best LLM (GPT-4o) gets only 61.5%, while weaker models can be as low as ~21%. 

CDEval – cultural dimensions benchmark.
- Questions aligned with Hofstede’s six cultural dimensions, covering multiple domains to probe value judgements (individualism vs collectivism, power distance, etc.).
- Constructed with GPT-4 generation and human verification; used to compare cultural attitudes of 17 LLMs. 

CAMeL / “Having Beer after Prayer?”
- Resource of 628 prompts and 20k entities to expose Arab vs Western cultural bias in multilingual LMs, mostly via story generation, infilling and NER tasks.
- Shows that models often “default” to Western cultural norms and stereotypes even when asked about Arab contexts. 

Culturally-Aware Conversations
- A conversational benchmark where LLMs must adapt style, politeness and framing to users from different cultures, not just answer trivia questions. 

These works all support the same main conclusion: current LLMs are far from culturally fair or knowledgeable, especially for under-represented regions. That is exactly the motivation for SemEval-2026 Task 7.


2.2 Typical approaches used so far

Most papers on BLEnD, CulturalBench, etc. do evaluation rather than training new models. The common recipe is:

- Prompt existing LLMs
Zero-shot: directly ask the question in natural language and have the model answer (short answer or pick an option).
Few-shot: include several Q-A demos in the prompt for the same region before asking a new question.

- Simple baselines
Random choice (25% accuracy).
Majority-answer baseline: choose the most frequent option for that region (or globally).
English-only baseline: when there are multilingual variants, always answer as if the user were from the US/UK, which tests bias directly.

- Model families evaluated
Large proprietary models: GPT-4, GPT-4o, Claude, Gemini etc.
Open-source LLMs (LLaMA-2/3, Mistral, xLlama, etc.) in different sizes.
In BLEnD and CulturalBench, GPT-4 family models are consistently best, but still far below humans and much weaker for Global South regions. 

- More advanced methods (recent work)
Retrieval-augmented generation (RAG): retrieve web content or Wikipedia snippets filtered by region and feed them into the LLM before answering.
In-context “salad bowl” prompts: mix demonstrations from multiple cultures and explicitly ask the model to “think as a person from region X”. (Used e.g. in follow-up work around BLEnD). 

Re-ranking approaches: cross-encoder model that scores (question, option, region) triples and picks the best-scoring option.



3. Proposed Approach

3.1. Overview

The implemented approach is a prompt-based + probabilistic scoring, using an autoregressive language model. For each question:

• A prompt is constructed that includes the cultural region.
• Each option is evaluated separately.
• The average score of the log-probabilities of the tokens in the option is calculated.
• The option with the maximum score is selected.


3.2. Prompt Construction

The prompt has the following general structure:

You are answering from the perspective of someone living in: {lang_reg}.
Choose the option that best matches everyday cultural norms and what is most typical there.
Question: {question}
Options:
A. …
B. …
C. …
Answer:

This formulation forces the model to:
• adopt an explicit cultural perspective,
• compare options based on “typical”, not “universal”.


3.3. Scoring Mechanism

For each option:
• the prompt and the option are concatenated,
• the model calculates the probability distribution for each token,
• the log-probability of the tokens in the option is extracted,
• the final score is the average of the log-probabilities.

Formal:

This method:
• penalizes improbable answers,
• avoids the random variations of free generation.


3.4. Architectural Diagram (description)

The architectural flow is as follows:

Input (question, options, lang_reg)
|
Prompt Builder
|
Prompt + Option (per option)
|
Language Model
|
Token Log-Probabilities
|
Mean Score per Option
|
Argmax Selection
|
Predicted Answer



4. Experimental Results

4.1. Datasets

We used:
• Training / Development set: provided by the SemEval organizers, containing questions with correct answers.
• Test set: unlabeled, used for the final evaluation.

The data is stored in TSV format and includes the following fields:
• index
• lang_reg
• question
• multiple_choice_options
• correct_answer (only for train/dev)

4.2. Exploratory Data Analysis

For SemEval-2026 Task 7, Track 2 we used the organisers’ pilot dataset trial_data_multiple_choice.tsv, which contains one row per multiple-choice question with four answer options. Each instance is labelled with a language–region code (lang_reg), the question text, the option list and the correct answer. After loading the file with pandas we first checked data quality. The dataset showed very few missing values, which we removed for the question and correct_answer columns, while imputing a small number of missing lang_reg entries with a special “unknown” label. By parsing the multiple_choice_options field we confirmed that almost all questions have exactly four options, as required by the task specification; the very few outliers and cases where the correct answer did not appear among the options were treated as annotation errors and removed. The dataset covers up to 23 language–region codes, with (e.g. some English and Spanish varieties being more frequent than others).



 We engineered simple numeric features such as question and answer length, and inspected their distributions globally and per region. Questions are generally short, but some regions tend to have slightly longer phrasings. No clear problematic outliers were found after manual inspection, so we kept the full cleaned dataset for modelling. Overall, the EDA confirmed that the pilot data is well-structured and suitable for building baseline models for cultural multiple-choice question answering.



The exploratory analysis shows that most questions are relatively short, with a unimodal distribution centered around roughly 6–10 words and a long right tail reaching ~25+ words. In contrast, correct answers are typically very concise: the majority are 1–2 words, with only a small number extending up to ~10 words. When broken down by language–region, question length varies moderately—most groups cluster around similar medians, but a few regions exhibit wider spreads and occasional outliers (both unusually short and unusually long questions), indicating some stylistic differences in how prompts are phrased across locales. Finally, the correlation analysis suggests that surface-level length features are largely independent: question length and answer length are almost uncorrelated (≈0.03), implying that longer questions do not systematically lead to longer answers in this dataset.























The Multinomial Naive Bayes baseline achieves moderate performance on the test set (37 examples), with accuracy ≈ 0.595, macro-precision ≈ 0.516, and macro-recall ≈ 0.565. The per-class report shows strong variability across language–region labels: for some classes (e.g., en-AU, en-GB, eu-ES, fa-IR, ko-KR, ta-LK) the model performs very well (sometimes reaching F1 = 1.0), while for others (e.g., el-GR, es-ES, es-MX, ga-IE, id-ID, ja-JP, zh-CN, zh-SG) recall and F1 drop to zero, indicating the model fails to retrieve instances of those classes. The confusion matrix suggests mistakes occur mainly between culturally or linguistically similar regions with overlapping vocabulary, and the gap between macro and weighted averages reflects imbalance and very small class support (typically 1–2 examples per label), making the per-class scores unstable and highly sensitive to a few incorrect predictions.


















The Multinomial Logistic Regression baseline performs better than Naive Bayes on this split, reaching an accuracy of ~0.68 (0.6757) with macro-precision ~0.63 and macro-recall ~0.67 (macro F1 ~0.62). The per-class report still shows high variance across language–region labels: several classes achieve strong results (many with precision/recall near 1.0, e.g., en-AU, en-GB, es-EC, es-MX, eu-ES, fa-IR, ko-KR, ta-LK), while a few remain challenging with zero recall/F1 (e.g., el-GR, es-ES, ga-IE, id-ID, ja-JP, zh-SG). The confusion matrix suggests that misclassifications tend to occur between culturally or linguistically similar regions, and the small support per class (often 1–2 samples) makes these per-label scores unstable—small changes in a few predictions can noticeably shift the metrics.















4.3. Experimental Settings

• Model: pre-trained Qwen/Qwen2.5-1.5B-Instruct model
• Fine-tuning: not used
• Decoding: probabilistic scoring (no sampling)
• Hardware: GPU (Colab)
• Batching: evaluation per instance

4.4. Evaluation Metrics

Primary metric:
• Accuracy – percentage of correct answers selected.
Secondary metrics:
• accuracy per cultural region,
• qualitative analysis of errors.


4.5. Results


system
split
metrics
score
Random 4 options
trial/pilot
accuracy
0.25
Qwen2.5-1.5B-Instruct
trial/pilot
accuracy
0.2973



In the trial/pilot split, we first establish a simple baseline by randomly selecting one of the four answer options, which yields the expected 25% accuracy. Using our Qwen2.5-1.5B-Instruct likelihood-based MCQ solver improves performance slightly to an overall accuracy of 0.2973 (macro accuracy over language–region: 0.2919). Performance varies substantially across lang_reg groups: the system achieves higher accuracies for some regions (e.g., en-AU ≈ 0.71, id-ID ≈ 0.60, bg-BG/ja-JP ≈ 0.57, es-EC ≈ 0.50), while several others remain near chance or below (e.g., fr-FR ≈ 0.13, multiple regions around 0.14–0.20). This spread suggests that cultural specificity and region-dependent conventions impact the model unevenly, and that a small general-purpose LLM without fine-tuning still struggles to consistently select culturally appropriate answers across all locales.















4.6. Discussion
The trial/pilot experiments show that the likelihood-based MCQ scoring with Qwen2.5-1.5B-Instruct provides a small improvement over the random-choice baseline (0.297 vs. 0.25 accuracy). While the overall gain is modest, the method is deterministic and reproducible, since predictions are obtained by comparing option log-likelihoods rather than sampling. Performance varies substantially across language–region groups, with some regions achieving noticeably higher accuracies while others remain close to chance, suggesting that cultural specificity is unevenly captured by a small general-purpose model.
Limitations. The approach is sensitive to prompt formulation and provides no explicit mechanism to adapt to difficult or low-performing regions. In addition, inference is relatively expensive because the model must score each candidate option separately (i.e., multiple forward passes per question), which increases runtime compared to single-pass generation.

5. Conclusion and Future Work
We presented a simple zero-shot baseline for SemEval Task 7 (Track 2, MCQ) based on prompt conditioning with the target region and probabilistic option scoring using an instruction-tuned LLM. Despite the absence of fine-tuning, the method captures some region-dependent preferences and slightly improves over random selection on the pilot split.
Future work includes (i) region-aware retrieval or lightweight adaptation methods to better handle difficult locales, (ii) prompt strategies tailored per region (or automatic prompt selection), and (iii) efficiency improvements such as batching, shared-prefix caching, or smaller scoring models to reduce computational cost.

6. References
BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages. Myung et al., NeurIPS 2024
CulturalBench: A Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs. Chiu et al., ACL 2025
Chivereanu, R., & Tufiș, D. (2025). RACAI at SemEval-2025 Task 7: Efficient adaptation of Large Language Models for Multilingual and Crosslingual Fact-Checked Claim Retrieval. SemEval-2025.

7. Links to the code
baseline-Task7
exploratoryAnalysis_task7



