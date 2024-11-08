# ICLR 2025 Submissions 



1. **WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct** [[pdf]](http://arxiv.org/abs/2308.09583) `ICLR 2025 Submission` (262 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          WizardMath is presented, which enhances the mathematical reasoning abilities of Llama-2, by applying the proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs), such as GPT-4, have shown remarkable performance in natural language processing (NLP) tasks, including challenging mathematical reasoning. However, most existing open-source models are only pre-trained on large-scale internet data and without math-related optimization. In this paper, we present WizardMath, which enhances the mathematical reasoning abilities of Llama-2, by applying our proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math. Through extensive experiments on two mathematical reasoning benchmarks, namely GSM8k and MATH, we reveal the extraordinary capabilities of our model. WizardMath surpasses all other open-source LLMs by a substantial margin. Furthermore, our model even outperforms ChatGPT-3.5, Claude Instant-1, PaLM-2 and Minerva on GSM8k, simultaneously surpasses Text-davinci-002, PaLM-1 and GPT-3 on MATH. More details and model weights are public at https://github.com/nlpxucan/WizardLM and https://huggingface.co/WizardLM.
     </details>

2. **Advancing LLM Reasoning Generalists with Preference Trees** [[pdf]](http://arxiv.org/abs/2404.02078) `ICLR 2025 Submission` (41 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Eurus, a suite of large language models (LLMs) optimized for reasoning, and derives a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Eurus, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, Eurus models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, Eurus-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. The strong performance of Eurus can be primarily attributed to UltraInteract, our newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. UltraInteract can be used in both supervised fine-tuning and preference learning. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. UltraInteract allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. Inspired by this, we derive a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
     </details>

3. **Common 7B Language Models Already Possess Strong Math Capabilities** [[pdf]](http://arxiv.org/abs/2403.04706) `ICLR 2025 Submission` (33 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy when selecting the best response from 256 random generations, and finds that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy of 97.7% and 72.0% on the GSM8K and MATH benchmarks, respectively, when selecting the best response from 256 random generations. The primary issue with the current base model is the difficulty in consistently eliciting its inherent mathematical capabilities. Notably, the accuracy for the first answer drops to 49.5% and 7.9% on the GSM8K and MATH benchmarks, respectively. We find that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers. However, the potential for extensive scaling is constrained by the scarcity of publicly available math questions. To overcome this limitation, we employ synthetic data, which proves to be nearly as effective as real data and shows no clear saturation when scaled up to approximately one million samples. This straightforward approach achieves an accuracy of 82.6% on GSM8K and 40.6% on MATH using LLaMA-2 7B models, surpassing previous models by 14.2% and 20.8%, respectively. We also provide insights into scaling behaviors across different reasoning complexities and error types.
     </details>

4. **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** [[pdf]](http://arxiv.org/abs/2407.21787) `ICLR 2025 Submission` (21 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Inference compute is explored as another axis for scaling by increasing the number of generated samples across multiple tasks and models, observing that coverage - the fraction of problems solved by any attempt - scales with the number of samples over four orders of magnitude.
     </details>


     <details>
          <summary>Abstract</summary>
          Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit the amount of compute to only one attempt per problem. Here, we explore inference compute as another axis for scaling by increasing the number of generated samples. Across multiple tasks and models, we observe that coverage - the fraction of problems solved by any attempt - scales with the number of samples over four orders of magnitude. In domains like coding and formal proofs, where all answers can be automatically verified, these increases in coverage directly translate into improved performance. When we apply repeated sampling to SWE-bench Lite, the fraction of issues solved with DeepSeek-V2-Coder-Instruct increases from 15.9% with one sample to 56% with 250 samples, outperforming the single-attempt state-of-the-art of 43% which uses more capable frontier models. Moreover, using current API pricing, amplifying the cheaper DeepSeek model with five samples is more cost-effective and solves more issues than paying a premium for one sample from GPT-4o or Claude 3.5 Sonnet. Interestingly, the relationship between coverage and the number of samples is often log-linear and can be modelled with an exponentiated power law, suggesting the existence of inference-time scaling laws. Finally, we find that identifying correct samples out of many generations remains an important direction for future research in domains without automatic verifiers. When solving math word problems from GSM8K and MATH, coverage with Llama-3 models grows to over 95% with 10,000 samples. However, common methods to pick correct solutions from a sample collection, such as majority voting or reward models, plateau beyond several hundred samples and fail to fully scale with the sample budget.
     </details>

5. **LiveBench: A Challenging, Contamination-Free LLM Benchmark** [[pdf]](http://arxiv.org/abs/2406.19314) `ICLR 2025 Submission` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a new benchmark for LLMs designed to be immune to both test set contamination and the pitfalls of LLM judging and human crowdsourcing, and releases LiveBench, the first benchmark that contains frequently-updated questions from recent information sources, and scores answers automatically according to objective ground-truth values.
     </details>


     <details>
          <summary>Abstract</summary>
          Test set contamination, wherein test data from a benchmark ends up in a newer model's training set, is a well-documented obstacle for fair LLM evaluation and can quickly render benchmarks obsolete. To mitigate this, many recent benchmarks crowdsource new prompts and evaluations from human or LLM judges; however, these can introduce significant biases, and break down when scoring hard questions. In this work, we introduce a new benchmark for LLMs designed to be immune to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We release LiveBench, the first benchmark that (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. To achieve this, LiveBench contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets, and it contains harder, contamination-free versions of tasks from previous benchmarks such as Big-Bench Hard, AMPS, and IFEval. We evaluate many prominent closed-source models, as well as dozens of open-source models ranging from 0.5B to 110B in size. LiveBench is difficult, with top models achieving below 65% accuracy. We release all questions, code, and model answers. Questions will be added and updated on a monthly basis, and we will release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future. We welcome community engagement and collaboration for expanding the benchmark tasks and models.
     </details>

6. **Training Language Models to Self-Correct via Reinforcement Learning** [[pdf]](http://arxiv.org/abs/2409.12917) `ICLR 2025 Submission` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A multi-turn online reinforcement learning (RL) approach that significantly improves an LLM's self-correction ability using entirely self-generated data, and uses appropriate regularization to steer the learning process into learning a self-correction behavior that is effective at test time as opposed to fitting high-reward responses for a given prompt.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Existing approaches for training self-correction either require multiple models or rely on a more capable model or other forms of supervision. To this end, we develop a multi-turn online reinforcement learning (RL) approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data. To build SCoRe, we first show that variants of supervised fine-tuning (SFT) on offline model-generated correction traces are insufficient for instilling self-correction behavior. In particular, we observe that training via SFT either suffers from a distribution mismatch between the training data and the model's own responses or implicitly prefers only a certain mode of correction behavior that is often not effective at test time. SCoRe addresses these challenges by training under the model's own distribution of self-generated correction traces and using appropriate regularization to steer the learning process into learning a self-correction strategy that is effective at test time as opposed to simply fitting high-reward responses for a given prompt. This regularization prescribes running a first phase of RL on a base model to generate a policy initialization that is less susceptible to collapse and then using a reward bonus to amplify self-correction during training. When applied to Gemini 1.0 Pro and 1.5 Flash models, we find that SCoRe achieves state-of-the-art self-correction performance, improving the base models' self-correction by 15.6% and 9.1% respectively on the MATH and HumanEval benchmarks.
     </details>

7. **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs** [[pdf]](https://arxiv.org/abs/2406.18629v1) `ICLR 2025 Submission` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically, and observes that in DPO, self-generated data is more effective than data generated by humans or GPT-4, due to the latter's out-of-distribution nature.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning presents a significant challenge for Large Language Models (LLMs) due to the extensive and precise chain of reasoning required for accuracy. Ensuring the correctness of each reasoning step is critical. To address this, we aim to enhance the robustness and factuality of LLMs by learning from human feedback. However, Direct Preference Optimization (DPO) has shown limited benefits for long-chain mathematical reasoning, as models employing DPO struggle to identify detailed errors in incorrect answers. This limitation stems from a lack of fine-grained process supervision. We propose a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically. Additionally, we have developed a data construction pipeline for Step-DPO, enabling the creation of a high-quality dataset containing 10K step-wise preference pairs. We also observe that in DPO, self-generated data is more effective than data generated by humans or GPT-4, due to the latter's out-of-distribution nature. Our findings demonstrate that as few as 10K preference data pairs and fewer than 500 Step-DPO training steps can yield a nearly 3% gain in accuracy on MATH for models with over 70B parameters. Notably, Step-DPO, when applied to Qwen2-72B-Instruct, achieves scores of 70.8% and 94.0% on the test sets of MATH and GSM8K, respectively, surpassing a series of closed-source models, including GPT-4-1106, Claude-3-Opus, and Gemini-1.5-Pro. Our code, data, and models are available at https://github.com/dvlab-research/Step-DPO.
     </details>

8. **Building Math Agents with Multi-Turn Iterative Preference Learning** [[pdf]](https://arxiv.org/abs/2409.02392v1) `ICLR 2025 Submission` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences and includes multi-turn DPO and multi-turn KTO as specific implementations.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent studies have shown that large language models' (LLMs) mathematical problem-solving capabilities can be enhanced by integrating external tools, such as code interpreters, and employing multi-turn Chain-of-Thought (CoT) reasoning. While current methods focus on synthetic data generation and Supervised Fine-Tuning (SFT), this paper studies the complementary direct preference learning approach to further improve model performance. However, existing direct preference learning algorithms are originally designed for the single-turn chat task, and do not fully address the complexities of multi-turn reasoning and external tool integration required for tool-integrated mathematical reasoning tasks. To fill in this gap, we introduce a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences. This framework includes multi-turn DPO and multi-turn KTO as specific implementations. The effectiveness of our framework is validated through training of various language models using an augmented prompt set from the GSM8K and MATH datasets. Our results demonstrate substantial improvements: a supervised fine-tuned Gemma-1.1-it-7B model's performance increased from 77.5% to 83.9% on GSM8K and from 46.1% to 51.2% on MATH. Similarly, a Gemma-2-it-9B model improved from 84.1% to 86.3% on GSM8K and from 51.0% to 54.5% on MATH.
     </details>

9. **LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts** [[pdf]](https://arxiv.org/abs/2407.04973v1) `ICLR 2025 Submission` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          LogicVista is proposed, an evaluation benchmark that assesses the integrated logical reasoning capabilities of multimodal large language models (MLLMs) in Visual contexts using a sample of 448 multiple-choice questions.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose LogicVista, an evaluation benchmark that assesses the integrated logical reasoning capabilities of multimodal large language models (MLLMs) in Visual contexts. Recent advancements in MLLMs have demonstrated various fascinating abilities, from crafting poetry based on an image to performing mathematical reasoning. However, there is still a lack of systematic evaluation of MLLMs' proficiency in logical reasoning tasks, which are essential for activities like navigation and puzzle-solving. Thus we evaluate general logical cognition abilities across 5 logical reasoning tasks encompassing 9 different capabilities, using a sample of 448 multiple-choice questions. Each question is annotated with the correct answer and the human-written reasoning behind the selection, enabling both open-ended and multiple-choice evaluation. A total of 8 MLLMs are comprehensively evaluated using LogicVista. Code and Data Available at https://github.com/Yijia-Xiao/LogicVista.
     </details>

10. **To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning** [[pdf]](https://arxiv.org/abs/2409.12183v1) `ICLR 2025 Submission` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks, and suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) via prompting is the de facto method for eliciting reasoning capabilities from large language models (LLMs). But for what kinds of tasks is this extra ``thinking'' really helpful? To analyze this, we conducted a quantitative meta-analysis covering over 100 papers using CoT and ran our own evaluations of 20 datasets across 14 models. Our results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks. On MMLU, directly generating the answer without CoT leads to almost identical accuracy as CoT unless the question or model's response contains an equals sign, indicating symbolic operations and reasoning. Following this finding, we analyze the behavior of CoT on these problems by separating planning and execution and comparing against tool-augmented LLMs. Much of CoT's gain comes from improving symbolic execution, but it underperforms relative to using a symbolic solver. Our results indicate that CoT can be applied selectively, maintaining performance while saving inference costs. Furthermore, they suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.
     </details>

11. **Automated Design of Agentic Systems** [[pdf]](http://arxiv.org/abs/2408.08435) `ICLR 2025 Submission` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work forms a new research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways, and presents a simple yet effective algorithm named Meta Agent Search, which can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents.
     </details>


     <details>
          <summary>Abstract</summary>
          Researchers are investing substantial effort in developing powerful general-purpose agents, wherein Foundation Models are used as modules within agentic systems (e.g. Chain-of-Thought, Self-Reflection, Toolformer). However, the history of machine learning teaches us that hand-designed solutions are eventually replaced by learned solutions. We formulate a new research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways. We further demonstrate that there is an unexplored yet promising approach within ADAS where agents can be defined in code and new agents can be automatically discovered by a meta agent programming ever better ones in code. Given that programming languages are Turing Complete, this approach theoretically enables the learning of any possible agentic system: including novel prompts, tool use, control flows, and combinations thereof. We present a simple yet effective algorithm named Meta Agent Search to demonstrate this idea, where a meta agent iteratively programs interesting new agents based on an ever-growing archive of previous discoveries. Through extensive experiments across multiple domains including coding, science, and math, we show that our algorithm can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents. Importantly, we consistently observe the surprising result that agents invented by Meta Agent Search maintain superior performance even when transferred across domains and models, demonstrating their robustness and generality. Provided we develop it safely, our work illustrates the potential of an exciting new research direction toward automatically designing ever-more powerful agentic systems to benefit humanity.
     </details>

12. **Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist** [[pdf]](http://arxiv.org/abs/2407.08733) `ICLR 2025 Submission` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that if a model really understands a problem, it should be robustly and readily applied across a diverse array of tasks, and introduced MATHCHECK, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently.
     </details>


     <details>
          <summary>Abstract</summary>
          Exceptional mathematical reasoning ability is one of the key features that demonstrate the power of large language models (LLMs). How to comprehensively define and evaluate the mathematical abilities of LLMs, and even reflect the user experience in real-world scenarios, has emerged as a critical issue. Current benchmarks predominantly concentrate on problem-solving capabilities, presenting a substantial risk of model overfitting and fails to accurately measure the genuine mathematical reasoning abilities. In this paper, we argue that if a model really understands a problem, it should be robustly applied across a diverse array of tasks. To this end, we introduce MathCheck, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently. MathCheck includes multiple mathematical reasoning tasks and robustness tests to facilitate a comprehensive evaluation of both mathematical reasoning ability and behavior testing. Utilizing MathCheck, we develop MathCheck-GSM and MathCheck-GEO to assess math textual reasoning and multi-modal reasoning abilities, respectively, serving as upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K. We adopt MathCheck-GSM and MathCheck-GEO to evaluate 26 LLMs and 17 MLLMs. Our results demonstrate that while frontier LLMs like GPT-4o continue to excel in various abilities on the checklist, many other model families exhibit a significant decline. Further experiments indicate that, compared to traditional math benchmarks, MathCheck better reflects true mathematical abilities and represents mathematical intelligence more linearly, thereby supporting our design. Using MathCheck, we can efficiently conduct informative behavior analysis to deeply investigate models. Finally, we show that our checklist paradigm can easily extend to other reasoning tasks.
     </details>

13. **Process-Driven Autoformalization in Lean 4** [[pdf]](http://arxiv.org/abs/2406.01940) `ICLR 2025 Submission` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark for autoformalization for large language models designed to evaluate the autoformalization capabilities of large language models (LLMs), and introduces a model that leverages the precise feedback from Lean 4 compilers to enhance autoformalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization, the conversion of natural language mathematics into formal languages, offers significant potential for advancing mathematical reasoning. However, existing efforts are limited to formal languages with substantial online corpora and struggle to keep pace with rapidly evolving languages like Lean 4. To bridge this gap, we propose a new benchmark \textbf{Form}alization for \textbf{L}ean~\textbf{4} (\textbf{\name}) designed to evaluate the autoformalization capabilities of large language models (LLMs). This benchmark encompasses a comprehensive assessment of questions, answers, formal statements, and proofs. Additionally, we introduce a \textbf{P}rocess-\textbf{S}upervised \textbf{V}erifier (\textbf{PSV}) model that leverages the precise feedback from Lean 4 compilers to enhance autoformalization. Our experiments demonstrate that the PSV method improves autoformalization, enabling higher accuracy using less filtered training data. Furthermore, when fine-tuned with data containing detailed process information, PSV can leverage the data more effectively, leading to more significant improvements in autoformalization for Lean 4. Our dataset and code are available at \url{https://github.com/rookie-joe/PDA}.
     </details>

14. **Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning** [[pdf]](http://arxiv.org/abs/2406.14283) `ICLR 2025 Submission` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By casting multi-step reasoning of LLMs as a heuristic search problem, Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning is introduced, which can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated impressive capability in many natural language tasks. However, the auto-regressive generation process makes LLMs prone to produce errors, hallucinations and inconsistent statements when performing multi-step reasoning. In this paper, by casting multi-step reasoning of LLMs as a heuristic search problem, we aim to alleviate the pathology by introducing Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning. By learning a plug-and-play Q-value model as heuristic function for estimating expected future rewards, our Q* can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task, which avoids the significant computational overhead and potential risk of performance degeneration on other tasks. Extensive experiments on GSM8K, MATH and MBPP demonstrate the superiority of our method, contributing to improving the reasoning performance of existing open-source LLMs.
     </details>

15. **Beyond Imitation: Learning Key Reasoning Steps from Dual Chain-of-Thoughts in Reasoning Distillation** [[pdf]](http://arxiv.org/abs/2405.19737) `ICLR 2025 Submission` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning is proposed, and it is found that EDIT benefits more from logical errors than from knowledge or mathematical calculation errors in dual CoTs.
     </details>


     <details>
          <summary>Abstract</summary>
          As Large Language Models (LLMs) scale up and gain powerful Chain-of-Thoughts (CoTs) reasoning abilities, practical resource constraints drive efforts to distill these capabilities into more compact Smaller Language Models (SLMs). We find that CoTs consist mainly of simple reasoning forms, with a small proportion ($\approx 4.7\%$) of key reasoning steps that truly impact conclusions. However, previous distillation methods typically involve supervised fine-tuning student SLMs only on correct CoTs data produced by teacher LLMs, resulting in students struggling to learn the key reasoning steps, instead imitating the teacher's reasoning forms and making errors or omissions on these steps. To address these issues, drawing an analogy to human learning, where analyzing mistakes according to correct solutions often reveals the crucial steps leading to successes or failures, we propose mistak\textbf{E}-\textbf{D}riven key reason\textbf{I}ng step distilla\textbf{T}ion (\textbf{EDIT}), a novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning. Firstly, to expose these crucial steps in CoTs, we design specific prompts to generate dual CoTs data with similar reasoning paths but divergent conclusions. Then, we apply the minimum edit distance algorithm on the dual CoTs data to locate these key steps and optimize the likelihood of these steps. Extensive experiments validate the effectiveness of EDIT across both in-domain and out-of-domain benchmark reasoning datasets. Further analysis shows that EDIT can generate high-quality CoTs with more correct key reasoning steps. Notably, we also explore how different mistake patterns affect performance and find that EDIT benefits more from logical errors than from knowledge or mathematical calculation errors in dual CoTs\footnote{Code can be found at \url{https://github.com/C-W-D/EDIT}}.
     </details>

16. **MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation** [[pdf]](http://arxiv.org/abs/2312.17080) `ICLR 2025 Submission` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel evaluation paradigm for Large Language Models (LLMs) that compels them to transition from a traditional question-answering role, akin to a student, to a solution-scoring role, akin to a teacher, and develops the MR-GSM8K benchmark.
     </details>


     <details>
          <summary>Abstract</summary>
          In this work, we introduce a novel evaluation paradigm for Large Language Models (LLMs) that compels them to transition from a traditional question-answering role, akin to a student, to a solution-scoring role, akin to a teacher. This paradigm, focusing on "reasoning about reasoning," hence termed meta-reasoning, shifts the emphasis from result-oriented assessments, which often neglect the reasoning process, to a more comprehensive evaluation that effectively distinguishes between the cognitive capabilities of different models. By applying this paradigm in the GSM8K dataset, we have developed the MR-GSM8K benchmark. Our extensive analysis includes several state-of-the-art models from both open-source and commercial domains, uncovering fundamental deficiencies in their training and evaluation methodologies. Notably, while models like Deepseek-v2 and Claude3-Sonnet closely competed with GPT-4 in GSM8K, their performance disparities expanded dramatically in MR-GSM8K, with differences widening to over 20 absolute points, underscoring the significant challenge posed by our meta-reasoning approach.
     </details>

17. **Prover-Verifier Games improve legibility of LLM outputs** [[pdf]](http://arxiv.org/abs/2407.13692) `ICLR 2025 Submission` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A training algorithm inspired by Prover-Verifier Game is proposed, which finds that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training, and that legibility training transfers to time-constrained humans tasked with verifying solution correctness.
     </details>


     <details>
          <summary>Abstract</summary>
          One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.
     </details>

18. **First-Step Advantage: Importance of Starting Right in Multi-Step Math Reasoning** [[pdf]](http://arxiv.org/abs/2311.07945) `ICLR 2025 Submission` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that getting the right start can lead to significant performance gains across all models, and proposed QuestCoT, where a smaller model first asks itself how to start, before proceeding with a chain of reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models can solve complex reasoning tasks better by learning to generate rationales for their predictions. Often these models know how to solve a task but their auto-regressive decoding nature leads to incorrect results if they start incorrectly. We observe that smaller models in particular when corrected, can solve a task that they would have otherwise struggled with. We demonstrate this phenomenon by using a larger model to guide smaller models, which leads to significantly improved performance (up to +24 points on the GSM8K dataset by 7B models). To assist smaller models in initiating the starting step, we propose QuestCoT, where a smaller model first asks itself how to start, before proceeding with a chain of reasoning. On various multistep mathematical reasoning datasets over multiple smaller models, we show that getting the right start can lead to significant performance gains across all models (gains of up to +6 points on GSM8K, +9 on SVAMP, +5 on ASDiv, and +7 on MultiArith).
     </details>

19. **Inductive or Deductive? Rethinking the Fundamental Reasoning Abilities of LLMs** [[pdf]](https://arxiv.org/abs/2408.00114v2) `ICLR 2025 Submission` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel framework, SolverLearner, that enables LLMs to learn the underlying function of inductive reasoning, and reveals that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases.
     </details>


     <details>
          <summary>Abstract</summary>
          Reasoning encompasses two typical types: deductive reasoning and inductive reasoning. Despite extensive research into the reasoning capabilities of Large Language Models (LLMs), most studies have failed to rigorously differentiate between inductive and deductive reasoning, leading to a blending of the two. This raises an essential question: In LLM reasoning, which poses a greater challenge - deductive or inductive reasoning? While the deductive reasoning capabilities of LLMs, (i.e. their capacity to follow instructions in reasoning tasks), have received considerable attention, their abilities in true inductive reasoning remain largely unexplored. To investigate into the true inductive reasoning capabilities of LLMs, we propose a novel framework, SolverLearner. This framework enables LLMs to learn the underlying function (i.e., $y = f_w(x)$), that maps input data points $(x)$ to their corresponding output values $(y)$, using only in-context examples. By focusing on inductive reasoning and separating it from LLM-based deductive reasoning, we can isolate and investigate inductive reasoning of LLMs in its pure form via SolverLearner. Our observations reveal that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases. Surprisingly, despite their strong inductive reasoning abilities, LLMs tend to relatively lack deductive reasoning capabilities, particularly in tasks involving ``counterfactual'' reasoning.
     </details>

20. **Progress or Regress? Self-Improvement Reversal in Post-training** [[pdf]](https://openreview.net/forum?id=MG18DR2dAN) `ICLR 2025 Submission` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Findings indicate that current self-improvement practices through post-training are inadequate for equipping models to tackle more complex problems and underscore the necessity of critical evaluation metrics in discerning the progress or regress dichotomy for self-improving LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-improvement through post-training methods such as iterative preference learning has been acclaimed for enhancing the problem-solving capabilities (e.g., mathematical reasoning) of Large Language Models (LLMs) without human intervention. However, as exploration deepens, it becomes crucial to assess whether these improvements genuinely signify progress in solving more challenging problems or if they could lead to unintended regressions. To address this, we propose a comprehensive evaluative framework that goes beyond the superficial pass@1 metric to scrutinize the underlying enhancements of post-training paradigms for self-improvement. Through rigorous experimentation and analysis across diverse problem-solving tasks, the empirical results point out the phenomenon of \emph{self-improvement reversal}, where models showing improved performance across benchmarks will paradoxically exhibit declines in broader, essential capabilities, like output diversity and out-of-distribution (OOD) generalization. These findings indicate that current self-improvement practices through post-training are inadequate for equipping models to tackle more complex problems. Furthermore, they underscore the necessity of our critical evaluation metrics in discerning the \emph{progress or regress} dichotomy for self-improving LLMs.
     </details>

21. **MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in Large Language Models Using Odyssey Math Data** [[pdf]](http://arxiv.org/abs/2406.18321) `ICLR 2025 Submission` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is indicated that while LLMs perform well on routine and moderately difficult tasks, they face significant challenges with Olympiad-level problems and complex university-level questions, particularly with the most demanding problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have significantly advanced natural language understanding and demonstrated strong problem-solving abilities. Despite these successes, most LLMs still struggle with solving mathematical problems due to the intricate reasoning required. This paper investigates the mathematical problem-solving capabilities of LLMs using the newly developed "MathOdyssey" dataset. The dataset includes diverse mathematical problems at high school and university levels, created by experts from notable institutions to rigorously test LLMs in advanced problem-solving scenarios and cover a wider range of subject areas. By providing the MathOdyssey dataset as a resource to the AI community, we aim to contribute to the understanding and improvement of AI capabilities in complex mathematical problem-solving. We conduct benchmarking on open-source models, such as Llama-3 and DBRX-Instruct, and closed-source models from the GPT series and Gemini models. Our results indicate that while LLMs perform well on routine and moderately difficult tasks, they face significant challenges with Olympiad-level problems and complex university-level questions. Our analysis shows a narrowing performance gap between open-source and closed-source models, yet substantial challenges remain, particularly with the most demanding problems. This study highlights the ongoing need for research to enhance the mathematical reasoning of LLMs. The dataset, results, and code are publicly available.
     </details>

22. **$Staple$: Towards Reliable Problem Solving with Large Language Models via Plan Optimization and Tree Search** [[pdf]](https://openreview.net/forum?id=P8FS9byr1c) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) exhibit the ability to perform step-by-step reasoning when tackling complex problems across various tasks. To improve the reliability of multi-step reasoning and mitigate potential hallucinations, sophisticated prompting techniques have been developed to provide instructions on $what$ $to$ $do$ at each step, offering reasoning guidance before addressing specific questions. However, this additional prompting can increase time and token consumption without guaranteeing effectiveness. In response, this paper proposes $Staple$, a novel plan retrieval augmented reasoning framework that utilizes offline plan optimization. This approach involves constructing a plan database of general-purpose reasoning instructions. Subsequently, online plan searching facilitates the direct retrieval of optimal and effective step-by-step plans from the database when addressing new questions, serving as guidance for LLMs to derive correct answers. The offline stage uses LLMs to self-generate and optimize plans, storing them as tree structures via Monte Carlo Tree Search (MCTS) to form the plan database. Extensive experiments on mathematical and multi-task problems show that $Staple$ achieves competitive problem-solving rates while minimizing token usage and interactions. Importantly, the plan trees in the database are human-interpretable, revealing the prioritization of various plan combinations for a given task. In addition, the plan database can be reused, updated, and expanded by users for a wider range of applications.
     </details>

23. **$\texttt{CLR-Bench}$: Evaluating Large Language Models in College-Level Reasoning** [[pdf]](https://openreview.net/forum?id=ToVvoHpk4L) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated their remarkable performance across various language understanding tasks. While emerging benchmarks have been proposed to evaluate LLMs in various domains such as mathematics and computer science, they merely measure the accuracy in terms of the final prediction on multi-choice questions. However, it remains insufficient to verify the essential understanding of LLMs given a chosen choice. To fill this gap, we present $\texttt{CLR-Bench}$ to comprehensively evaluate the LLMs in complex college-level reasoning. Specifically, $(i)$ we prioritize 16 challenging college disciplines in computer science and artificial intelligence. The dataset contains 5 types of questions, while each question is associated with detailed explanations from experts. $(ii)$ To quantify a fair evaluation of LLMs' reasoning ability, we formalize the criteria with two novel metrics. Q$\rightarrow$A is utilized to measure the performance of direct **a**nswer prediction, and Q$\rightarrow$AR effectively considers the joint ability to **a**nswer the question and provide **r**ationale simultaneously. Extensive experiments are conducted with 40 LLMs over 1,018 discipline-specific questions. The results demonstrate the key insights that LLMs, even the best closed-source LLM, i.e., GPT-4 turbo, tends to '***guess***' the college-level answers. It shows a dramatic decrease in accuracy from 63.31\% Q$\rightarrow$A to 39.00\% Q$\rightarrow$AR, indicating an unsatisfactory reasoning ability.
     </details>

24. **3D-Prover: Diversity Driven Theorem Proving With Determinantal Point Processes** [[pdf]](http://arxiv.org/abs/2410.11133) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A key challenge in automated formal reasoning is the intractable search space, which grows exponentially with the depth of the proof. This branching is caused by the large number of candidate proof tactics which can be applied to a given goal. Nonetheless, many of these tactics are semantically similar or lead to an execution error, wasting valuable resources in both cases. We address the problem of effectively pruning this search, using only synthetic data generated from previous proof attempts. We first demonstrate that it is possible to generate semantically aware tactic representations which capture the effect on the proving environment, likelihood of success and execution time. We then propose a novel filtering mechanism which leverages these representations to select semantically diverse and high quality tactics, using Determinantal Point Processes. Our approach, 3D-Prover, is designed to be general, and to augment any underlying tactic generator. We demonstrate the effectiveness of 3D-Prover on the miniF2F-valid and miniF2F-test benchmarks by augmenting the ReProver LLM. We show that our approach leads to an increase in the overall proof rate, as well as a significant improvement in the tactic success rate, execution time and diversity.
     </details>

25. **A Graph-Based Synthetic Data Pipeline for Scaling High-Quality Data** [[pdf]](https://openreview.net/forum?id=CEE9cAQJ10) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Synthesizing high-quality data for continual training has been proven to be effective for boosting the performance of Large Language Models (LLMs). However, previous approaches struggle to easily scale up synthetic data and incur high costs in creating high-quality data. In this paper, we propose the Graph-based Synthetic Data Pipeline (GSDP), an economical and scalable framework for high-quality data synthesis. Inspired by knowledge graphs, we extracted knowledge points from seed data and constructed a knowledge points relationships graph to explore their interconnections. By exploring the implicit relationships among knowledge, our method achieves $\times$255 data expansion. Furthermore, GSDP led by open-source models, achieves synthesis quality comparable to GPT-4-0613 while maintaining $\times$100 lower costs. To tackle the most challenging mathematical reasoning task, we present the GSDP-MATH dataset comprising over 1.91 million pairs of math problems and answers. After fine-tuning on GSDP-MATH, GSDP-7B achieves 37.7\% accuracy on MATH and 78.4\% on GSM8K, demonstrating the effectiveness of our method. The dataset and models trained in this paper will be available.
     </details>

26. **Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation** [[pdf]](https://arxiv.org/abs/2410.02725v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance and enabling more efficient and scalable compute utilization during inference for LLMs is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Inference-time computation is a powerful paradigm to enhance the performance of large language models (LLMs), with Best-of-N sampling being a widely used technique. However, this method is computationally expensive, requiring both (1) an external reward model and (2) the generation of multiple samples. In this work, we introduce a new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance. We use a generative reward model formulation, allowing the LLM to predict mid-generation the probability that restarting the generation will yield a better response. These predictions are obtained without an external reward model and can be used to decide whether or not to generate more samples, prune unpromising samples early on, or to pick the best sample. This capability is very inexpensive as it involves generating a single predefined token. Trained using a dataset constructed with real unfiltered LMSYS user prompts, Llama 3.1 8B's win rate against GPT-4 on AlpacaEval increases from 21% to 34% with 16 samples and math performance on GSM8K improves from 84% to 91%. By sampling only when the LLM determines that it is beneficial to do so and adaptively adjusting temperature annealing, we demonstrate that 74% of the improvement from using 16 samples can be achieved with only 1.2 samples on average. We further demonstrate that 50-75% of samples can be pruned early in generation with minimal degradation in performance. Overall, our methods enable more efficient and scalable compute utilization during inference for LLMs.
     </details>

27. **Advancing Mathematical Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages** [[pdf]](https://openreview.net/forum?id=GtpubstM1D) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Advancements in large language models (LLMs) have significantly expanded their capabilities across various domains. However, mathematical reasoning remains a challenging area, prompting the development of math-specific LLMs such as LLEMMA, DeepSeekMath, and Qwen2-Math, among others. These models typically follow a two-stage training paradigm: pre-training with math-related corpora and post-training with problem datasets for supervised fine-tuning (SFT). Despite these efforts, the improvements in mathematical reasoning achieved through continued pre-training (CPT) are often less significant compared to those obtained via SFT. This study addresses this discrepancy by exploring alternative strategies during the pre-training phase, focusing on the use of problem-solving data over general mathematical corpora. We investigate three primary research questions: (1) Can problem-solving data enhance the model's mathematical reasoning capabilities more effectively than general mathematical corpora during CPT? (2) Are synthetic data from the same source equally effective, and which synthesis methods are most efficient? (3) How do the capabilities developed from the same problem-solving data differ between the CPT and SFT stages, and what factors contribute to these differences? Our findings indicate that problem-solving data significantly enhances the model's mathematical capabilities compared to general mathematical corpora. We also identify effective data synthesis methods, demonstrating that the tutorship amplification synthesis method achieves the best performance. Furthermore, while SFT facilitates instruction-following abilities, it underperforms compared to CPT with the same data, which can be partially attributed to its poor learning capacity for hard multi-step problem-solving data. These insights provide valuable guidance for optimizing the mathematical reasoning capabilities of LLMs, culminating in our development of a powerful mathematical base model called JiuZhang-8B.
     </details>

28. **Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation** [[pdf]](https://openreview.net/forum?id=7NL74jUiMg) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Formal proofs are challenging to write even for experienced experts. Recent progress in Neural Theorem Proving (NTP) shows promise in expediting this process. However, the formal corpora available on the Internet are limited compared to the general text, posing a significant data scarcity challenge for NTP. To address this issue, this work proposes Alchemy, a general framework for data synthesis that constructs formal theorems through symbolic mutation. Specifically, for each candidate theorem in Mathlib, we identify all invocable theorems that can be used to rewrite or apply to it. Subsequently, we mutate the candidate theorem by replacing the corresponding term in the statement with its equivalent form or antecedent. As a result, our method increases the number of theorems in Mathlib by an order of magnitude, from 110k to 6M. Furthermore, we perform continual pretraining and supervised finetuning on this augmented corpus for large language models. Experimental results demonstrate the effectiveness of our approach, achieving a 5% absolute performance improvement on Leandojo benchmark. Additionally, our synthetic data achieve a 2.5% absolute performance gain on the out-of-distribution miniF2F benchmark. To provide further insights, we conduct a comprehensive analysis of synthetic data composition and the training paradigm, offering valuable guidance for developing a strong theorem prover.
     </details>

29. **AoPS Dataset: Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation** [[pdf]](https://openreview.net/forum?id=Bgz3okeZ7H) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Advances in Large Language Models (LLMs) have sparked interest in their ability to solve Olympiad-level math problems.  However, the training and evaluation of these models are constrained by the limited size and quality of available datasets, as creating large-scale data for such advanced problems requires extensive effort from human experts. In addition, current benchmarks are prone to contamination, leading to unreliable evaluations. In this paper, we present an automated pipeline that leverages the rich resources of the Art of Problem Solving (AoPS) forum, which predominantly features Olympiad-level problems and community-driven solutions. Using open-source LLMs, we develop a method to extract question-answer pairs from the forum, resulting in **AoPS-Instruct**, a dataset of more than 650,000 high-quality QA pairs. Our experiments demonstrate that fine-tuning LLMs on AoPS-Instruct improves their reasoning abilities across various benchmarks.  Moreover, we build an automatic pipeline that introduces **LiveAoPSBench**, an evolving evaluation set with timestamps, derived from the latest forum data, providing a contamination-resistant benchmark for assessing LLM performance. Notably, we observe a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure rather than true reasoning ability.  Our work presents a scalable approach to creating and maintaining large-scale, high-quality datasets for advanced math reasoning, offering valuable insights into the capabilities and limitations of LLMs in this domain.  Our benchmark is available at [livemathbench.github.io/leaderboard](https://livemathbench.github.io/leaderboard).
     </details>

30. **Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count** [[pdf]](http://arxiv.org/abs/2410.15787) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Transformers often struggle with length generalization, meaning they fail to generalize to sequences longer than those encountered during training. While arithmetic tasks are commonly used to study length generalization, certain tasks are considered notoriously difficult, e.g., multi-operand addition (requiring generalization over both the number of operands and their lengths) and multiplication (requiring generalization over both operand lengths). In this work, we achieve approximately 2-3x length generalization on both tasks, which is the first such achievement in arithmetic Transformers. We design task-specific scratchpads enabling the model to focus on a fixed number of tokens per each next-token prediction step, and apply multi-level versions of Position Coupling (Cho et al., 2024; McLeish et al., 2024) to let Transformers know the right position to attend to. On the theory side, we prove that a 1-layer Transformer using our method can solve multi-operand addition, up to operand length and operand count that are exponential in embedding dimension.
     </details>

31. **Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics** [[pdf]](http://arxiv.org/abs/2410.21272) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Do large language models (LLMs) solve reasoning tasks by learning robust generalizable algorithms, or do they memorize training data? To investigate this question, we use arithmetic reasoning as a representative task. Using causal analysis, we identify a subset of the model (a circuit) that explains most of the model's behavior for basic arithmetic logic and examine its functionality. By zooming in on the level of individual circuit neurons, we discover a sparse set of important neurons that implement simple heuristics. Each heuristic identifies a numerical input pattern and outputs corresponding answers. We hypothesize that the combination of these heuristic neurons is the mechanism used to produce correct arithmetic answers. To test this, we categorize each neuron into several heuristic types-such as neurons that activate when an operand falls within a certain range-and find that the unordered combination of these heuristic types is the mechanism that explains most of the model's accuracy on arithmetic prompts. Finally, we demonstrate that this mechanism appears as the main source of arithmetic accuracy early in training. Overall, our experimental results across several LLMs show that LLMs perform arithmetic using neither robust algorithms nor memorization; rather, they rely on a "bag of heuristics".
     </details>

32. **AutoCode4Math: Learning Autonomous Code Integration for Math LLMs** [[pdf]](https://openreview.net/forum?id=QhjosARfay) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent research on tool integration for math Large Language Models (LLMs)  aims to combine complementary strengths of chain-of-thought (CoT) reasoning and code execution. However, we discover a critical limitation: current tool-integrated math LLMs rely on externally dictated instructions to decide whether to use CoT or code, lacking the autonomy to choose the most appropriate method independently. This prompts us to study \emph{Autonomous Code integration } (AutoCode) for math LLMs, which enables models to \emph{independently} develop their own methodology-selection strategy in the absence of reliable supervision. To address this challenge, we propose an innovative Expectation-Maximization (EM) formulation that refines the model's decision-making through the exploration of its capabilities. This framework alternates between (a) computing a reference strategy that improves the model's belief over its capabilities through self-exploration, and (b) updating the model based on the refined belief. We further enhance this framework with an efficient implementation, incorporating a novel data synthesis strategy and off-policy reinforcement learning. Extensive experiments demonstrate that our approach, using only a public query set, significantly boosts the performance of existing math LLMs, raising accuracy by nearly 20% to 65.28%  on the challenging MATH benchmark on the MATH benchmark, while reducing code executions by up to 65% .
     </details>

33. **Automatic Curriculum Expert Iteration for Reliable LLM Reasoning** [[pdf]](http://arxiv.org/abs/2410.07627) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Automatic Curriculum Expert Iteration is proposed to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them--and SOTA baselines are compared, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness.
     </details>


     <details>
          <summary>Abstract</summary>
          Hallucinations (i.e., generating plausible but inaccurate content) and laziness (i.e. excessive refusals or defaulting to "I don't know") persist as major challenges in LLM reasoning. Current efforts to reduce hallucinations primarily focus on factual errors in knowledge-grounded tasks, often neglecting hallucinations related to faulty reasoning. Meanwhile, some approaches render LLMs overly conservative, limiting their problem-solving capabilities. To mitigate hallucination and laziness in reasoning tasks, we propose Automatic Curriculum Expert Iteration (Auto-CEI) to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them. In our method, Expert Iteration explores the reasoning trajectories near the LLM policy, guiding incorrect paths back on track to reduce compounding errors and improve robustness; it also promotes appropriate "I don't know" responses after sufficient reasoning attempts. The curriculum automatically adjusts rewards, incentivizing extended reasoning before acknowledging incapability, thereby pushing the limits of LLM reasoning and aligning its behaviour with these limits. We compare Auto-CEI with various SOTA baselines across logical reasoning, mathematics, and planning tasks, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness.
     </details>

34. **B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners** [[pdf]](https://openreview.net/forum?id=P6dwZJpJ4m) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In the absence of extensive human-annotated data for complex reasoning tasks, self-improvement -- where models are trained on their own outputs -- has emerged as a primary method for enhancing performance. Recently, the approach to self-improvement has shifted toward a more dynamic, online fashion through iterative training processes. However, the critical factors underlying the mechanism of these self-improving methods remain poorly understood, such as under what conditions self-improvement is effective, and what are the bottlenecks in the current iterations. In this work, we identify and propose methods to monitor two pivotal factors in this iterative process: (1) the model's ability to explore and generate high-quality responses among multiple candidates (exploration); and (2) the reliability of external rewards in selecting the best responses from the generated outputs (exploitation). These factors are inherently moving targets throughout the self-improvement cycles, yet their dynamics are rarely discussed in prior research -- It remains unclear what impedes continual model enhancement after only a few iterations.  Using mathematical reasoning as a case study, we begin with a quantitative analysis to track the dynamics of exploration and exploitation, discovering that a model's exploratory capabilities rapidly deteriorate over iterations, and the effectiveness of exploiting external rewards diminishes as well due to shifts in distribution from the original policy. Motivated by these findings, we introduce B-STaR, a Self-Taught Reasoning framework that autonomously adjusts configurations across iterations to Balance exploration and exploitation, thereby optimizing the self-teaching effectiveness based on the current policy model and available rewards. Our experiments in mathematical reasoning demonstrate that B-STaR not only enhances the model's exploratory capabilities throughout training but also achieves a more effective balance between exploration and exploitation, leading to superior performance. Crucially, this work deconstructs the opaque nature of self-training algorithms, elucidating the interpretable dynamics throughout the process and highlighting current limitations for future research to address.
     </details>

35. **BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search** [[pdf]](http://arxiv.org/abs/2409.17972) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel approach, BEATS, to enhance mathematical problem-solving abilities of large Language Models, which leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited exceptional performance across a broad range of tasks and domains. However, they still encounter difficulties in solving mathematical problems due to the rigorous and logical nature of mathematics. Previous studies have employed techniques such as supervised fine-tuning (SFT), prompt engineering, and search-based methods to improve the mathematical problem-solving abilities of LLMs. Despite these efforts, their performance remains suboptimal and demands substantial computational resources. To address this issue, we propose a novel approach, BEATS, to enhance mathematical problem-solving abilities. Our method leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps. Additionally, we introduce a new back-verification technique that uses LLMs to validate the correctness of the generated answers. Furthermore, we employ a pruning tree search to optimize search time while achieving strong performance. Notably, our method improves Qwen2-7b-Instruct's score from 36.94 to 61.52, outperforming GPT4's 42.5 on the MATH benchmark.
     </details>

36. **CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search** [[pdf]](https://openreview.net/forum?id=VQwI055flA) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advancements in neural theorem proving integrate large language models with tree search algorithms like Monte Carlo Tree Search (MCTS), where the language model suggests tactics and the tree search finds the complete proof path. However, many tactics proposed by the language model converge to semantically or strategically similar, reducing diversity and increasing search costs by expanding redundant proof paths. This issue exacerbates as computation scales and more tactics are explored per state. Furthermore, the trained value function suffers from false negatives, label imbalance, and domain gaps due to biased data construction.  To address these challenges, we propose CARTS (diversified tactic CAlibration and bias-Resistant Tree Search), which balances tactic diversity and importance while calibrating model confidence. CARTS also introduce preference modeling and an adjustment term related to the ratio of valid tactics to improve the bias-resistance of the value function. Experimental results demonstrate that CARTS consistently outperforms previous methods achieving a pass@l rate of 49.6\% on the miniF2F-test benchmark. Further analysis confirms that CARTS improves tactic diversity and leads to a more balanced tree search.
     </details>

37. **CPL: Critical Planning Step Learning Boosts LLM Generalization in Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.08642) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes searching within the action space on high-level abstract plans to enhance model generalization and introduces Critical Plan Step Learning (CPL), comprising: searching on plan, using Monte Carlo Tree Search (MCTS) to explore diverse plan steps in multi-step reasoning tasks, and learning critical plan steps through Step-level Advantage Preference Optimization (Step-APO), which integrates advantage estimates for step preference obtained via MCTS into Direct Preference Optimization (DPO).
     </details>


     <details>
          <summary>Abstract</summary>
          Post-training large language models (LLMs) to develop reasoning capabilities has proven effective across diverse domains, such as mathematical reasoning and code generation. However, existing methods primarily focus on improving task-specific reasoning but have not adequately addressed the model's generalization capabilities across a broader range of reasoning tasks. To tackle this challenge, we introduce Critical Planning Step Learning (CPL), which leverages Monte Carlo Tree Search (MCTS) to explore diverse planning steps in multi-step reasoning tasks. Based on long-term outcomes, CPL learns step-level planning preferences to improve the model's planning capabilities and, consequently, its general reasoning capabilities. Furthermore, while effective in many scenarios for aligning LLMs, existing preference learning approaches like Direct Preference Optimization (DPO) struggle with complex multi-step reasoning tasks due to their inability to capture fine-grained supervision at each step. We propose Step-level Advantage Preference Optimization (Step-APO), which integrates an advantage estimate for step-level preference pairs obtained via MCTS into the DPO. This enables the model to more effectively learn critical intermediate planning steps, thereby further improving its generalization in reasoning tasks. Experimental results demonstrate that our method, trained exclusively on GSM8K and MATH, not only significantly improves performance on GSM8K (+10.5%) and MATH (+6.5%), but also enhances out-of-domain reasoning benchmarks, such as ARC-C (+4.0%), BBH (+1.8%), MMLU-STEM (+2.2%), and MMLU (+0.9%).
     </details>

38. **Can LLMs Solve Long Math Word Problems Better?** [[pdf]](https://openreview.net/forum?id=C9ju8QQSCv) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
     </details>

39. **Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks** [[pdf]](http://arxiv.org/abs/2410.01428) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Critic-guided planning with Retrieval-augmentation, CR-Planner is introduced, a novel framework that leverages fine-tuned critic models to guide both reasoning and retrieval processes through planning that significantly outperforms baselines on challenging domain-knowledge-intensive and reasoning-heavy tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          State-of-the-art large language models (LLMs) exhibit impressive problem-solving capabilities but may struggle with complex reasoning and factual correctness. Existing methods harness the strengths of chain-of-thought and retrieval-augmented generation (RAG) to decompose a complex problem into simpler steps and apply retrieval to improve factual correctness. These methods work well on straightforward reasoning tasks but often falter on challenging tasks such as competitive programming and mathematics, due to frequent reasoning errors and irrelevant knowledge retrieval. To address this, we introduce Critic-guided planning with Retrieval-augmentation, CR-Planner, a novel framework that leverages fine-tuned critic models to guide both reasoning and retrieval processes through planning. CR-Planner solves a problem by iteratively selecting and executing sub-goals. Initially, it identifies the most promising sub-goal from reasoning, query generation, and retrieval, guided by rewards given by a critic model named sub-goal critic. It then executes this sub-goal through sampling and selecting the optimal output based on evaluations from another critic model named execution critic. This iterative process, informed by retrieved information and critic models, enables CR-Planner to effectively navigate the solution space towards the final answer. We employ Monte Carlo Tree Search to collect the data for training the critic models, allowing for a systematic exploration of action sequences and their long-term impacts. We validate CR-Planner on challenging domain-knowledge-intensive and reasoning-heavy tasks, including competitive programming, theorem-driven math reasoning, and complex domain retrieval problems. Our experiments demonstrate that CR-Planner significantly outperforms baselines, highlighting its effectiveness in addressing challenging problems by improving both reasoning and retrieval.
     </details>

40. **CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2410.02229) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs from publicly available high-quality source code, and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have made significant progress in natural language understanding and generation, driven by scalable pretraining and advanced finetuning. However, enhancing reasoning abilities in LLMs, particularly via reinforcement learning from human feedback (RLHF), remains challenging due to the scarcity of high-quality preference data, which is labor-intensive to annotate and crucial for reward model (RM) finetuning. To alleviate this issue, we introduce CodePMP, a scalable preference model pretraining (PMP) pipeline that utilizes a large corpus of synthesized code-preference pairs from publicly available high-quality source code. CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs. We evaluate CodePMP on mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor, LogiQA2.0), consistently showing significant improvements in reasoning performance of LLMs and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
     </details>

41. **CogMath: Evaluating LLMs' Authentic Mathematical Ability from a Cognitive Perspective** [[pdf]](https://openreview.net/forum?id=x1nlO1d1iG) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          As large language models (LLMs) exhibit potential in solving complex mathematical tasks, increasing attention has been directed toward constructing benchmarks to evaluate their mathematical capabilities. However, existing benchmarks are either limited to specific task types (e.g., long-text problem understanding) or rely solely on a coarse measure of answer accuracy, making them insufficient for assessing a model's authentic mathematical proficiency. In this paper, we propose CogMath, which provides a comprehensive assessment of LLMs' mathematical abilities based on human cognitive processes. Specifically, inspired by cognitive theories, CogMath formalizes the reasoning process into 3 stages that align with human cognition: problem comprehension, problem solving, and solution summarization, and encompasses 9 fine-grained evaluation dimensions from perspectives such as numerical calculation, knowledge, and counterfactuals. In each dimension, to carry out a scientific evaluation, we develop an ``Inquiry-Judge-Reference'' multi-agent system, where the Inquiry agent generates inquiries that assess LLMs' mastery from this dimension, the Judge agent ensures the inquiry quality, and the Reference agent provides correct responses for comparison with the LLMs' actual performances. A LLM is considered to truly master a problem only when excelling in all inquiries from the 9 dimensions. In experiments, we evaluate 7 mainstream LLMs by applying CogMath to three benchmarks, which cover the full K-12 mathematical curriculum. The results reveal that the authentic mathematical capabilities of current LLMs are overestimated by 30-40%. Moreover, we locate their strengths and weaknesses across different stages/dimensions, offering constructive insights to further enhance their reasoning abilities.
     </details>

42. **Conformal Language Model Reasoning with Coherent Factuality** [[pdf]](https://openreview.net/forum?id=AJpUZd8Clb) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language models are increasingly being used in important decision pipelines, so ensuring the correctness of their outputs is crucial. Recent work has proposed evaluating the “factuality” of claims decomposed from a language model generation and applying conformal prediction techniques to filter out those claims that are not factual. This can be effective for tasks such as information retrieval, where constituent claims may be evaluated in isolation for factuality, but is not appropriate for reasoning tasks, as steps of a logical argument can be evaluated for correctness only within the context of the claims that have preceded them. To capture this, we define “coherent factuality” and develop a conformal-prediction-based method to guarantee coherent factuality of language model outputs. Our approach applies split conformal prediction to subgraphs within a ``deducibility" graph that we construct to represent the steps of a reasoning problem. We evaluate our method on mathematical reasoning problems from the MATH and FELM datasets, and find that our algorithm achieves coherent factuality across target coverage levels, consistently producing orderings of correct claims that are substantiated by previous ones. Moreover, we achieve 90\% factuality on our stricter definition while retaining 80\% or more of the original claims, highlighting the utility of our deducibility-graph-guided approach.
     </details>

43. **CreDes: Causal Reasoning Enhancement and Dual-End Searching for Solving Long-Range Reasoning Problems using LLMs** [[pdf]](http://arxiv.org/abs/2410.01696) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By integrating CRE and DES (CreDes), the model has realized simultaneous multi-step reasoning, circumventing the inefficiencies from cascading multiple one-step reasoning like the Chain-of-Thought (CoT).
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated limitations in handling combinatorial optimization problems involving long-range reasoning, partially due to causal hallucinations and huge search space. As for causal hallucinations, i.e., the inconsistency between reasoning and corresponding state transition, this paper introduces the Causal Relationship Enhancement (CRE) mechanism combining cause-effect interventions and the Individual Treatment Effect (ITE) to guarantee the solid causal rightness between each step of reasoning and state transition. As for the long causal range and huge search space limiting the performances of existing models featuring single-direction search, a Dual-End Searching (DES) approach is proposed to seek solutions by simultaneously starting from both the initial and goal states on the causal probability tree. By integrating CRE and DES (CreDes), our model has realized simultaneous multi-step reasoning, circumventing the inefficiencies from cascading multiple one-step reasoning like the Chain-of-Thought (CoT). Experiments demonstrate that CreDes significantly outperforms existing State-Of-The-Art (SOTA) solutions in long-range reasoning tasks in terms of both accuracy and time efficiency.
     </details>

44. **Critic-CoT: Boosting the reasoning abilities of large language model via Chain-of-Thought Critic** [[pdf]](https://openreview.net/forum?id=JEehcb48Vp) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Self-critic has become a crucial mechanism for enhancing the reasoning performance of LLMs. However, current approaches mainly involve basic prompts for intuitive instance-level feedback, which resembles System-1 processes and limits the reasoning capabilities. Moreover, there is a lack of in-depth investigations into the relationship between LLM's ability to criticize and its task-solving performance. To address these issues, we propose Critic-CoT, a novel framework that pushes LLMs toward System-2-like critic capability. Through a step-wise CoT reasoning paradigm and the automatic construction of distant-supervision data without human annotation, Critic-CoT enables LLMs to engage in slow, analytic self-critique and refinement, thereby improving their reasoning abilities. Experiments on GSM8K and MATH demonstrate that our enhanced model significantly boosts task-solving performance by filtering out invalid solutions or iterative refinement. Furthermore, we investigate the intrinsic correlation between critique and task-solving abilities within LLMs, discovering that these abilities can mutually reinforce each other rather than conflict.
     </details>

45. **DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search** [[pdf]](http://arxiv.org/abs/2410.03864) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          DOTS is proposed, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM.
     </details>


     <details>
          <summary>Abstract</summary>
          Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.
     </details>

46. **Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model** [[pdf]](https://arxiv.org/abs/2410.03136v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP), which incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Enhancing the reasoning capabilities of large language models (LLMs) remains a key challenge, especially for tasks that require complex, multi-step decision-making. Humans excel at these tasks by leveraging deliberate planning with an internal world model to simulate the potential outcomes of various actions. Inspired by this, we propose a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP). Unlike previous approaches that rely solely on Chain-of-Thought (CoT) reasoning in natural language, SWAP incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps. Moreover, SWAP overcomes the challenge of accurate world state predictions in complex reasoning tasks by introducing a Generator-Discriminator architecture, which enables more reliable world modeling. Specifically, the generator predicts the next state, and the discriminator ensures alignment with the logical consistency required by the problem context. SWAP also encourages the policy model to explore a broad range of potential actions to prevent premature convergence. By resolving the bottlenecks of generation diversity for both actions and states using diversity-based modeling (DBM) and improving discrimination accuracy through contrastive ranking (CR), SWAP significantly enhances the reasoning performance of LLMs. We evaluate SWAP across diverse reasoning-intensive benchmarks including math reasoning, logical reasoning, and coding tasks. Extensive experiments demonstrate that SWAP achieves substantial improvements over the baselines and consistently outperforms existing LLMs of similar sizes.
     </details>

47. **Detecting Problematic Questions to Support Math Word Problem Design** [[pdf]](https://openreview.net/forum?id=ma4SUzeCLR) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          When designing math word problems, teachers must ensure the clarity and precision of the question to avoid multiple interpretations and unanswerable situations, thereby maintaining consistent grading standards and effectiveness. We address these issues to provide comprehensive support to teachers in creating clear, solvable, and formal math word problems. In this paper, we present MathError, a dataset of real-world math word problems annotated with error types to investigate the need for question correction. Our work explores how large language models (LLMs) can assist teachers in detecting problematic questions to support math word problem design in scenarios with limited data, simulating real-world conditions with minimal training samples. Preliminary results demonstrate the models' capabilities in detecting problematic questions and identify areas for further research and development in educational applications.
     </details>

48. **Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces** [[pdf]](http://arxiv.org/abs/2410.09918) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Dualformer is a single Transformer model that seamlessly integrates both the fast and slow reasoning modes, and achieves improved performance with LLM fine-tuning, showing its generalization beyond task-specific models.
     </details>


     <details>
          <summary>Abstract</summary>
          In human cognition theory, human thinking is governed by two systems: the fast and intuitive System 1 and the slower but more deliberative System 2. Recent studies have shown that incorporating System 2 process into Transformers including large language models (LLMs), significantly enhances their reasoning capabilities. Nevertheless, models that purely resemble System 2 thinking require substantially higher computational costs and are much slower to respond. To address this challenge, we present Dualformer, a single Transformer model that seamlessly integrates both the fast and slow reasoning modes. Dualformer is obtained by training on data with randomized reasoning traces, where different parts of the traces are dropped during training. The dropping strategies are specifically tailored according to the trace structure, analogous to analyzing our thinking process and creating shortcuts with patterns. At inference time, our model can be configured to output only the solutions (fast mode) or both the reasoning chain and the final solution (slow mode), or automatically decide which mode to engage (auto mode). In all cases, Dualformer outperforms the corresponding baseline models in both performance and computational efficiency: (1) in slow mode, Dualformer optimally solves unseen 30 x 30 maze navigation tasks 97.6% of the time, surpassing the Searchformer (trained on data with complete reasoning traces) baseline performance of 93.3%, while only using 45.5% fewer reasoning steps; (2) in fast mode, Dualformer completes those tasks with an 80% optimal rate, significantly outperforming the Solution-Only model (trained on solution-only data), which has an optimal rate of only 30%. For math problems, our techniques have also achieved improved performance with LLM fine-tuning, showing its generalization beyond task-specific models.
     </details>

49. **DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models** [[pdf]](http://arxiv.org/abs/2411.00836) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The rapid advancements in Vision-Language Models (VLMs) have shown great potential in tackling mathematical reasoning tasks that involve visual context. Unlike humans who can reliably apply solution steps to similar problems with minor modifications, we found that SOTA VLMs like GPT-4o can consistently fail in these scenarios, revealing limitations in their mathematical reasoning capabilities. In this paper, we investigate the mathematical reasoning robustness in VLMs and evaluate how well these models perform under different variants of the same question, such as changes in visual numerical values or function graphs. While several vision-based math benchmarks have been developed to assess VLMs' problem-solving capabilities, these benchmarks contain only static sets of problems and cannot easily evaluate mathematical reasoning robustness. To fill this gap, we introduce DynaMath, a dynamic visual math benchmark designed for in-depth assessment of VLMs. DynaMath includes 501 high-quality, multi-topic seed questions, each represented as a Python program. Those programs are carefully designed and annotated to enable the automatic generation of a much larger set of concrete questions, including many different types of visual and textual variations. DynaMath allows us to evaluate the generalization ability of VLMs, by assessing their performance under varying input conditions of a seed question. We evaluated 14 SOTA VLMs with 5,010 generated concrete questions. Our results show that the worst-case model accuracy, defined as the percentage of correctly answered seed questions in all 10 variants, is significantly lower than the average-case accuracy. Our analysis emphasizes the need to study the robustness of VLMs' reasoning abilities, and DynaMath provides valuable insights to guide the development of more reliable models for mathematical reasoning.
     </details>

50. **Dynamic Alignment of Representations for Enhanced Chain-of-Thought Reasoning in Large Language Models** [[pdf]](https://openreview.net/forum?id=7hRuaiRlgZ) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Representations encode rich semantic information, implying that editing them could serve as a effective tool (i.e., DAS, REFT) for parameter-efficient finetuning (PEFT). However, existing approaches typically focus on general categories of representations or selecting an appropriate number of continuous representations for each datasets, which limits their adaptability and performance. In contrast, our method dynamically selects representations requiring intervention at the instance level, referred to as misaligned representations, which are characterized by a lack of semantic information or appropriate attention. Identifying these misaligned representations poses challenging, as they serve different roles in varying contexts. It is evident that crucial representations, which are those that primarily receive information flow from themselves or significantly influence other representations, are likely to encompass misaligned representations. Consequently, we simplify the task by pivot our focus to crucial representations and aim to accurately locate them. We adaptively update crucial representation amidst uncertainty, freezing the base model while learning an updated direction for each layer. Involving both identification and updating of representations, we present a PEFT method, termed Dynamic Alignment of Representations (DAR). We validate the effectiveness of our method on eight diverse datasets across two scenarios, arithmetic and commonsense, and three base models: LLaMA-2-7B, LLaMA-2-13B, and LLaMA-3-8B. Notably, our method yields improvements of 17.47% and 3.11% over LLaMA-2-7B and ReFT on the GSM8K dataset, respectively. Additionally, it requires only 51 times fewer parameters than LoRA, demonstrating significant parameter efficiency. Furthermore, our method can be easily extended to few-shot learning.
     </details>

51. **Embedding Self-Correction as an Inherent Ability in Large Language Models for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.10735) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel mechanism, the Chain of Self-Correction (CoSC), specifically designed to embed self-correction as an inherent ability in LLMs, enabling them to validate and rectify their own results.
     </details>


     <details>
          <summary>Abstract</summary>
          Accurate mathematical reasoning with Large Language Models (LLMs) is crucial in revolutionizing domains that heavily rely on such reasoning. However, LLMs often encounter difficulties in certain aspects of mathematical reasoning, leading to flawed reasoning and erroneous results. To mitigate these issues, we introduce a novel mechanism, the Chain of Self-Correction (CoSC), specifically designed to embed self-correction as an inherent ability in LLMs, enabling them to validate and rectify their own results. The CoSC mechanism operates through a sequence of self-correction stages. In each stage, the LLMs generate a program to address a given problem, execute this program using program-based tools to obtain an output, subsequently verify this output. Based on the verification, the LLMs either proceed to the next correction stage or finalize the answer. This iterative self-correction process allows the LLMs to refine their reasoning steps and improve the accuracy of their mathematical reasoning. To enable the CoSC mechanism at a low cost, we employ a two-phase finetuning approach. In the first phase, the LLMs are trained with a relatively small volume of seeding data generated from GPT-4, establishing an initial CoSC capability. In the second phase, the CoSC capability is further enhanced by training with a larger volume of self-generated data using the trained model in the first phase, without relying on the paid GPT-4. Our comprehensive experiments demonstrate that CoSC significantly improves performance on traditional mathematical datasets among existing open-source LLMs. Notably, our CoSC-Code-34B model achieved a 53.5% score on MATH, the most challenging mathematical reasoning dataset in the public domain, surpassing the performance of well-established models such as ChatGPT, GPT-4, and even multi-modal LLMs like GPT-4V, Gemini-1.0 Pro, and Gemini-1.0 Ultra.
     </details>

52. **Emergent properties with repeated examples** [[pdf]](http://arxiv.org/abs/2410.07041) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that for a fixed number of training steps, models trained on smaller sets of repeated examples outperform models trained on larger sets of single-use examples and that two-set training provides for faster learning and better performance.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common divisor, modular multiplication, and matrix eigenvalues, we show that for a fixed number of training steps, models trained on smaller sets of repeated examples outperform models trained on larger sets of single-use examples. We also demonstrate that two-set training - repeated use of a small random subset of examples, along normal sampling on the rest of the training set - provides for faster learning and better performance. This highlights that the benefits of repetition can outweigh those of data diversity. These datasets and problems provide a controlled setting to shed light on the still poorly understood interplay between generalization and memorization in deep learning.
     </details>

53. **Enhancing Mathematical Reasoning in Language Models Through Focused Differentiation Training** [[pdf]](https://openreview.net/forum?id=vuuYbA1vB2) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Enhancing the mathematical capabilities of large language models (LLMs) is crucial for applications requiring precise and rigorous mathematical reasoning. Current models, even when trained with methods like Direct Preference Optimization (DPO), often struggle to effectively differentiate between correct and erroneous mathematical responses, especially when errors occur in multi-step solutions. Traditional approaches focusing on token or logit-level analysis fail to capture the nuanced semantic differences in mathematical reasoning. To address this challenge, we propose leveraging the rich semantic information embedded in the hidden state space of LLMs. Our novel approach, Focused Differentiation Training (FDT), fine-tunes the model by emphasizing the differences between the hidden states of correct and incorrect responses, rather than their common features. Unlike other methods that detect errors at the token or logits level and often rely on human input or more powerful models, our approach enhances mathematical reasoning capabilities using only the model's inherent abilities. This methodology promotes a more accurate alignment with mathematical correctness, thereby improving the model's ability to evaluate and generate precise mathematical responses. Experimental results demonstrate that our algorithm substantially outperforms traditional alignment methods in mathematical tasks, offering a robust solution for enhancing the mathematical reasoning capabilities of language models.
     </details>

54. **Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization** [[pdf]](http://arxiv.org/abs/2410.09302) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Direct Q-function Optimization (DQO), which formulates the response generation process as a Markov Decision Process (MDP) and utilizes the soft actor-critic (SAC) framework to optimize a Q-function directly parameterized by the language model.
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement Learning (RL) plays a crucial role in aligning large language models (LLMs) with human preferences and improving their ability to perform complex tasks. However, current approaches either require significant computational resources due to the use of multiple models and extensive online sampling for training (e.g., PPO) or are framed as bandit problems (e.g., DPO, DRO), which often struggle with multi-step reasoning tasks, such as math problem-solving and complex reasoning that involve long chains of thought. To overcome these limitations, we introduce Direct Q-function Optimization (DQO), which formulates the response generation process as a Markov Decision Process (MDP) and utilizes the soft actor-critic (SAC) framework to optimize a Q-function directly parameterized by the language model. The MDP formulation of DQO offers structural advantages over bandit-based methods, enabling more effective process supervision. Experimental results on two math problem-solving datasets, GSM8K and MATH, demonstrate that DQO outperforms previous methods, establishing it as a promising offline reinforcement learning approach for aligning language models.
     </details>

55. **ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection** [[pdf]](https://arxiv.org/abs/2410.04509v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ErrorRadar is introduced, the first benchmark designed to assess MLLMs' capabilities in such a task, and contains 2,500 high-quality multimodal K-12 mathematical problems, collected from real-world student interactions in an educational organization, with rigorous annotation and rich metadata such as problem type and error category.
     </details>


     <details>
          <summary>Abstract</summary>
          As the field of Multimodal Large Language Models (MLLMs) continues to evolve, their potential to revolutionize artificial intelligence is particularly promising, especially in addressing mathematical reasoning tasks. Current mathematical benchmarks predominantly focus on evaluating MLLMs' problem-solving ability, yet there is a crucial gap in addressing more complex scenarios such as error detection, for enhancing reasoning capability in complicated settings. To fill this gap, we formally formulate the new task: multimodal error detection, and introduce ErrorRadar, the first benchmark designed to assess MLLMs' capabilities in such a task. ErrorRadar evaluates two sub-tasks: error step identification and error categorization, providing a comprehensive framework for evaluating MLLMs' complex mathematical reasoning ability. It consists of 2,500 high-quality multimodal K-12 mathematical problems, collected from real-world student interactions in an educational organization, with rigorous annotation and rich metadata such as problem type and error category. Through extensive experiments, we evaluated both open-source and closed-source representative MLLMs, benchmarking their performance against educational expert evaluators. Results indicate significant challenges still remain, as GPT-4o with best performance is still around 10% behind human evaluation. The dataset will be available upon acceptance.
     </details>

56. **Evaluating Robustness of Reward Models for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.01729) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Reward models are key in reinforcement learning from human feedback (RLHF) systems, aligning the model behavior with human preferences. Particularly in the math domain, there have been plenty of studies using reward models to align policies for improving reasoning capabilities. Recently, as the importance of reward models has been emphasized, RewardBench is proposed to understand their behavior. However, we figure out that the math subset of RewardBench has different representations between chosen and rejected completions, and relies on a single comparison, which may lead to unreliable results as it only see an isolated case. Therefore, it fails to accurately present the robustness of reward models, leading to a misunderstanding of its performance and potentially resulting in reward hacking. In this work, we introduce a new design for reliable evaluation of reward models, and to validate this, we construct RewardMATH, a benchmark that effectively represents the robustness of reward models in mathematical reasoning tasks. We demonstrate that the scores on RewardMATH strongly correlate with the results of optimized policy and effectively estimate reward overoptimization, whereas the existing benchmark shows almost no correlation. The results underscore the potential of our design to enhance the reliability of evaluation, and represent the robustness of reward model. We make our code and data publicly available.
     </details>

57. **Exchange of Perspective Prompting Enhances Reasoning in Large Language Models** [[pdf]](https://openreview.net/forum?id=jBBjZp0EVs) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have made significant advancements in addressing diverse natural language processing (NLP) tasks. However, their performance is often limited by inherent comprehension of problems. To address this limitation, we propose Exchange-of-Perspective (EoP), a novel framework designed to incorporate external perspectives by swapping answers for the same question presented with different definitions. We conducted extensive and comprehensive experiments on seven benchmarks. The results show that EoP can significantly improve performance. For instance, compared to the non-commutative baseline PHP, with GPT-3.5-Turbo and EoP, we observe a 3.6% improvement on AQuA (60.6% → 64.2%), while GPT-4-powered EoP achieves a 7.7% overall accuracy improvement on Math (53.9% → 61.6%).
     </details>

58. **Executing Arithmetic: Fine-Tuning Large Language Models as Turing Machines** [[pdf]](http://arxiv.org/abs/2410.07896) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Composable Arithmetic Execution Framework (CAEF) is proposed that enables LLMs to learn to execute step-by-step computations by emulating Turing Machines, thereby gaining a genuine understanding of computational logic.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing and reasoning tasks. However, their performance in the foundational domain of arithmetic remains unsatisfactory. When dealing with arithmetic tasks, LLMs often memorize specific examples rather than learning the underlying computational logic, limiting their ability to generalize to new problems. In this paper, we propose a Composable Arithmetic Execution Framework (CAEF) that enables LLMs to learn to execute step-by-step computations by emulating Turing Machines, thereby gaining a genuine understanding of computational logic. Moreover, the proposed framework is highly scalable, allowing composing learned operators to significantly reduce the difficulty of learning complex operators. In our evaluation, CAEF achieves nearly 100% accuracy across seven common mathematical operations on the LLaMA 3.1-8B model, effectively supporting computations involving operands with up to 100 digits, a level where GPT-4o falls short noticeably in some settings.
     </details>

59. **Exposing the Achilles' Heel: Evaluating LLMs Ability to Handle Mistakes in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.10834) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models, and highlights GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have been applied to Math Word Problems (MWPs) with transformative impacts, revolutionizing how these complex problems are approached and solved in various domains including educational settings. However, the evaluation of these models often prioritizes final accuracy, overlooking the crucial aspect of reasoning capabilities. This work addresses this gap by focusing on the ability of LLMs to detect and correct reasoning mistakes. We introduce a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models. Our comprehensive benchmarking reveals significant insights into the strengths and weaknesses of state-of-the-art models, such as GPT-4o, GPT-4, GPT-3.5Turbo, and others. We highlight GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models. Additionally, we identify issues related to data contamination and memorization, impacting the reliability of LLMs in real-world applications. Our findings emphasize the importance of rigorous evaluation of reasoning processes and propose future directions to enhance the generalization and robustness of LLMs in mathematical problem-solving.
     </details>

60. **FLARE: Faithful Logic-Aided Reasoning and Exploration** [[pdf]](http://arxiv.org/abs/2410.11900) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Faithful CoT (F-CoT) is introduced, a novel interpretable approach for traversing the problem space using task decompositions and it is demonstrated that model faithfulness positively correlates with overall performance and allows pinpointing the decisive factors sufficient for and leading to the correct answer with optimal reasoning during the multi-hop search.
     </details>


     <details>
          <summary>Abstract</summary>
          Modern Question Answering (QA) and Reasoning approaches based on Large Language Models (LLMs) commonly use prompting techniques, such as Chain-of-Thought (CoT), assuming the resulting generation will have a more granular exploration and reasoning over the question space and scope. However, such methods struggle with generating outputs that are faithful to the intermediate chain of reasoning produced by the model. On the other end of the spectrum, neuro-symbolic methods such as Faithful CoT (F-CoT) propose to combine LLMs with external symbolic solvers. While such approaches boast a high degree of faithfulness, they usually require a model trained for code generation and struggle with tasks that are ambiguous or hard to formalise strictly. We introduce $\textbf{F}$aithful $\textbf{L}$ogic-$\textbf{A}$ided $\textbf{R}$easoning and $\textbf{E}$xploration ($\textbf{FLARE}$), a novel interpretable approach for traversing the problem space using task decompositions. We use the LLM to plan a solution, soft-formalise the query into facts and predicates using a logic programming code and simulate that code execution using an exhaustive multi-hop search over the defined space. Our method allows us to compute the faithfulness of the reasoning process w.r.t. the generated code and analyse the steps of the multi-hop search without relying on external solvers. Our methods achieve SOTA results on $\mathbf{7}$ out of $\mathbf{9}$ diverse reasoning benchmarks. We also show that model faithfulness positively correlates with overall performance and further demonstrate that $\textbf{FLARE}$ allows pinpointing the decisive factors sufficient for and leading to the correct answer with optimal reasoning during the multi-hop search.
     </details>

61. **Fine-grained Hallucination Detection and Mitigation in Language Model Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.06304) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a comprehensive taxonomy that categorizes the common hallucinations in mathematical reasoning task into six types: fabrication, factual inconsistency, context inconsistency, instruction inconsistency, logical inconsistency, and logical error and proposes FG-PRM (Fine-Grained Process Reward Model), an augmented model designed to detect and mitigate hallucinations in a fine-grained, step-level manner.
     </details>


     <details>
          <summary>Abstract</summary>
          Hallucinations in large language models (LLMs) pose significant challenges in tasks requiring complex multi-step reasoning, such as mathematical problem-solving. Existing approaches primarily detect the presence of hallucinations but lack a nuanced understanding of their types and manifestations. In this paper, we first introduce a comprehensive taxonomy that categorizes the common hallucinations in mathematical reasoning task into six types: fabrication, factual inconsistency, context inconsistency, instruction inconsistency, logical inconsistency, and logical error. We then propose FG-PRM (Fine-Grained Process Reward Model), an augmented model designed to detect and mitigate hallucinations in a fine-grained, step-level manner. To address the limitations of manually labeling training data, we propose an automated method for generating fine-grained hallucination data using LLMs. By injecting hallucinations into reasoning steps of correct solutions, we create a diverse and balanced synthetic dataset for training FG-PRM, which consists of six specialized Process Reward Models (PRMs), each tailored to detect a specific hallucination type. Our FG-PRM demonstrates superior performance across two key tasks: 1) Fine-grained hallucination detection: classifying hallucination types for each reasoning step; and 2) Verification: ranking multiple LLM-generated outputs to select the most accurate solution, mitigating reasoning hallucinations. Our experiments show that FG-PRM outperforms ChatGPT-3.5 and Claude-3 on fine-grained hallucination detection and substantially boosts the performance of LLMs on GSM8K and MATH benchmarks.
     </details>

62. **FormalAlign: Automated Alignment Evaluation for Autoformalization** [[pdf]](https://openreview.net/forum?id=B5RrIFMqbe) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces \textsc{FormalAlign], the first automated framework designed for evaluating the alignment between natural and formal languages in autoformalization, and significantly reduces the need for manual verification.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization aims to convert informal mathematical proofs into machine-verifiable formats, bridging the gap between natural and formal languages. However, ensuring semantic alignment between the informal and formalized statements remains challenging. Existing approaches heavily rely on manual verification, hindering scalability. To address this, we introduce \textsc{FormalAlign}, the first automated framework designed for evaluating the alignment between natural and formal languages in autoformalization. \textsc{FormalAlign} trains on both the autoformalization sequence generation task and the representational alignment between input and output, employing a dual loss that combines a pair of mutually enhancing autoformalization and alignment tasks. Evaluated across four benchmarks augmented by our proposed misalignment strategies, \textsc{FormalAlign} demonstrates superior performance. In our experiments, \textsc{FormalAlign} outperforms GPT-4, achieving an Alignment-Selection Score 11.58\% higher on \forml-Basic (99.21\% vs. 88.91\%) and 3.19\% higher on MiniF2F-Valid (66.39\% vs. 64.34\%). This effective alignment evaluation significantly reduces the need for manual verification. Both the dataset and code can be accessed via~\url{https://github.com/rookie-joe/FormalAlign}.
     </details>

63. **GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models** [[pdf]](https://arxiv.org/abs/2410.05229v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GSM-Symbolic is introduced, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions that enables more controllable evaluations, providing key insights and more reliable metrics for measuring the reasoning capabilities of models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Large Language Models (LLMs) have sparked interest in their formal reasoning capabilities, particularly in mathematics. The GSM8K benchmark is widely used to assess the mathematical reasoning of models on grade-school-level questions. While the performance of LLMs on GSM8K has significantly improved in recent years, it remains unclear whether their mathematical reasoning capabilities have genuinely advanced, raising questions about the reliability of the reported metrics. To address these concerns, we conduct a large-scale study on several SOTA open and closed models. To overcome the limitations of existing evaluations, we introduce GSM-Symbolic, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions. GSM-Symbolic enables more controllable evaluations, providing key insights and more reliable metrics for measuring the reasoning capabilities of models.Our findings reveal that LLMs exhibit noticeable variance when responding to different instantiations of the same question. Specifically, the performance of all models declines when only the numerical values in the question are altered in the GSM-Symbolic benchmark. Furthermore, we investigate the fragility of mathematical reasoning in these models and show that their performance significantly deteriorates as the number of clauses in a question increases. We hypothesize that this decline is because current LLMs cannot perform genuine logical reasoning; they replicate reasoning steps from their training data. Adding a single clause that seems relevant to the question causes significant performance drops (up to 65%) across all state-of-the-art models, even though the clause doesn't contribute to the reasoning chain needed for the final answer. Overall, our work offers a more nuanced understanding of LLMs' capabilities and limitations in mathematical reasoning.
     </details>

64. **GeoILP: A Synthetic Dataset to Guide Large-Scale Rule Induction** [[pdf]](https://openreview.net/forum?id=cfGpIcOIa5) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Inductive logic programming (ILP) is a machine learning approach aiming to learn explanatory rules from data.     While existing ILP systems can successfully solve small-scale tasks, large-scale applications with various language biases are rarely explored.     Besides, it is crucial for a large majority of current ILP systems to require expert-defined language bias, which hampers the development of ILP towards broader utilizations.     In this paper, we introduce GeoILP, a large-scale synthetic dataset of diverse ILP tasks involving numerous aspects of language bias.     % including complex rule forms, high deduction complexity, and more realistic assumptions.     The ILP tasks are built from geometry problems, at the level from textbook exercise to regional International Mathematical Olympiad (IMO), with the help of a deduction engine.     These problems are elaborately selected to cover all challenging language biases, such as recursion, predicate invention, and high arity.     Experimental results show that no existing method can solve GeoILP tasks.     In addition, along with classic symbolic-form data, we provide image-form data to boost the development of the joint learning of neural perception and symbolic rule induction.
     </details>

65. **GeoMath: A Benchmark for Multimodal Mathematical Reasoning in Remote Sensing** [[pdf]](https://openreview.net/forum?id=i3aFjkfnXO) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Vision-language models (VLMs) have demonstrated impressive performance in various Earth observation tasks, particularly in zero-shot capabilities. However, their mathematical reasoning skills in remote sensing (RS) remain unexplored due to the lack of relevant data. To close this gap, we introduce \dataset, a multimodal mathematical reasoning benchmark meticulously designed for the RS domain. It comprises 3773 high-quality vehicle-related questions from aerial perspectives, spanning 6 mathematical subjects and 20 topics. All data used in this benchmark were collected by our drones from various altitudes and perspectives. Despite the limited geographical coverage, full access to all parameters of the RS images and detailed vehicle information ensures that the constructed mathematical problems are rigorous and diverse. With GeoMath, we have conducted a comprehensive and quantitative evaluation of 14 prominent VLMs. Solving these math problems requires high-resolution visual perception and domain-specific mathematical knowledge, which poses a challenge even for state-of-the-art VLMs. We further explore the impact of image resolution and the zero-shot prompting strategy on the scores, analyzing the reasons behind GPT-4o's reasoning errors. By comparing the gap between InternVL2 and GPT-4o, we find that the latter exhibits some level of cross-view knowledge transfer capability.
     </details>

66. **GeoX: Geometric Problem Solving Through Unified Formalized Vision-Language Pre-training** [[pdf]](https://openreview.net/forum?id=6RiBl5sCDF) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite their proficiency in general tasks, Multi-modal Large Language Models (MLLMs) struggle with automatic Geometry Problem Solving (GPS), which demands understanding diagrams, interpreting symbols, and performing complex reasoning. This limitation arises from their pre-training on natural images and texts, along with the lack of automated verification in the problem-solving process. Besides, current geometric specialists are limited by their task-specific designs, making them less effective for broader geometric problems. To this end, we present GeoX, a multi-modal large model focusing on geometric understanding and reasoning tasks. Given the significant differences between geometric diagram-symbol and natural image-text, we introduce unimodal pre-training to develop a diagram encoder and symbol decoder, enhancing the understanding of geometric images and corpora. Furthermore, we introduce geometry-language alignment, an effective pre-training paradigm that bridges the modality gap between unimodal geometric experts. We propose a Generator-And-Sampler Transformer (GS-Former) to generate discriminative queries and eliminate uninformative representations from unevenly distributed geometric signals. Finally, GeoX benefits from visual instruction tuning, empowering it to take geometric images and questions as input and generate verifiable solutions. Experiments show that GeoX outperforms both generalists and geometric specialists on publicly recognized benchmarks, such as GeoQA, UniGeo, Geometry3K, and PGPS9k. Our data and code will be released soon to accelerate future research on automatic GPS.
     </details>

67. **GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning** [[pdf]](http://arxiv.org/abs/2410.02203) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GraphIC is presented, a novel approach that leverages graph-based representations of reasoning processes, coupled with Bayesian Networks (BNs) to select ICEs, and outperforms both training-free and training-based models in selecting ICEs, excelling in terms of both effectiveness and efficiency.
     </details>


     <details>
          <summary>Abstract</summary>
          In-context learning (ICL) enables large language models (LLMs) to generalize to new tasks by incorporating a few in-context examples (ICEs) directly in the input, without updating parameters. However, the effectiveness of ICL heavily relies on the selection of ICEs, and conventional text-based embedding methods are often inadequate for tasks that require multi-step reasoning, such as mathematical and logical problem solving. This is due to the bias introduced by shallow semantic similarities that fail to capture the deeper reasoning structures required for these tasks. We present GraphIC, a novel approach that leverages graph-based representations of reasoning processes, coupled with Bayesian Networks (BNs) to select ICEs. Graph structures inherently filter out shallow semantics while preserving the core reasoning structure. Importantly, BNs capture the dependency of a node's attributes on its parent nodes, closely mirroring the hierarchical nature of human cognition-where each thought is shaped by preceding ones. This makes BNs particularly well-suited for multi-step reasoning tasks, aligning the process more closely with human-like reasoning. Extensive experiments across three types of reasoning tasks (mathematical reasoning, code generation, and logical reasoning) demonstrate that GraphIC outperforms both training-free and training-based models in selecting ICEs, excelling in terms of both effectiveness and efficiency. We show that GraphIC enhances ICL's performance and interoperability, significantly advancing ICE selection for multi-step reasoning tasks.
     </details>

68. **HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows** [[pdf]](http://arxiv.org/abs/2409.17433) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite recent advancements in large language models (LLMs), their performance on complex reasoning problems requiring multi-step thinking and combining various skills is still limited. To address this, we propose a novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner. Our approach consists of two key components: 1) a new approach for slow, deliberate reasoning called Dynamic Workflow, which automatically decomposes complex problems into more manageable sub-tasks and dynamically designs a workflow to assemble specialized LLM or symbolic reasoning tools to solve sub-tasks; 2) Hybrid Thinking, a general framework that dynamically combines fast and slow thinking based on problem complexity. Finally, we propose an easy-to-scale method for automatically synthesizing a large-scale dataset of 27K challenging reasoning problems for complex reasoning and a hybrid thinking tuning method that trains smaller LLMs on this dataset to internalize the fast/slow hybrid reasoning strategies. Experiments on four reasoning benchmark datasets demonstrate that our slow thinking with dynamic workflows significantly outperforms Chain-of-Thought, and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance. Fine-tuning using our hybrid thinking approach also significantly boosts the complex reasoning capabilities of open-source language models. The results showcase the promise of slow thinking, dynamic workflows, and hybrid thinking in expanding the frontier of complex problem-solving with LLMs\footnote{Code and data will be released at \url{https://github.com/wenlinyao/HDFlow}.}.
     </details>

69. **Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[pdf]](https://openreview.net/forum?id=I4YAIwrsXa) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Lean is an advanced proof assistant designed to facilitate formal theorem proving by providing a variety of interactive feedback. In this paper, we explore methodologies to leverage proof assistant feedback to augment the capabilities of large language models in constructing formal proofs. First, we deploy online reinforcement learning using Lean verification outcomes as the reward signal to improve the proof completion policy. This straightforward approach shows great promise in enhancing the model's alignment with the formal verification system. In addition, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. The tree structure is organized to represent the transitions of intermediate tactic states, extracted from the compilation messages given by Lean's tactic mode. The intrinsic reward is constructed to incentivize the discovery of novel tactic states, which helps to to mitigate the sparse-reward problem inherent in proof search. These techniques lead to a more efficient planning scheme for formal proof generation, achieving new state-of-the-art results on both miniF2F and ProofNet benchmarks.
     </details>

70. **Hint Marginalization for Improved Reasoning in Large Language Models** [[pdf]](https://openreview.net/forum?id=DzKdjWe59v) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited an impressive capability to perform reasoning tasks, especially if they are encouraged to generate a sequence of intermediate steps. Reasoning performance can be improved by suitably combining multiple LLM responses, generated either in parallel in a single query, or via sequential interactions with LLMs throughout the reasoning process. Existing strategies for combination, such as self-consistency and progressive-hint-prompting, make inefficient usage of the LLM responses. We present Hint Marginalization, a novel and principled algorithmic framework to enhance the reasoning capabilities of LLMs. Our approach can be viewed as an iterative sampling strategy for forming a Monte Carlo approximation of an underlying distribution of answers, with the goal of identifying the mode the most likely answer. Empirical evaluation on several benchmark datasets for arithmetic reasoning demonstrates the superiority of the proposed approach.
     </details>

71. **How Can Language Models Learn from Mistakes on Grade-School Math Problems** [[pdf]](https://openreview.net/forum?id=zpDGwcmMV4) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language models have demonstrated remarkable performance in solving reasoning tasks; however, even the strongest models still occasionally make reasoning mistakes. Recently, there has been active research aimed at improving reasoning accuracy, particularly by using pretrained language models to "self-correct'' their mistakes via multi-round prompting. In this paper, we follow this line of work but focus on understanding the usefulness of incorporating ``error-correction'' data directly into the pretraining stage. This data consists of erroneous solution steps immediately followed by their corrections. Using a synthetic math dataset, we show promising results: this type of pretrain data can help language models achieve higher reasoning accuracy directly (i.e., through simple auto-regression, without multi-round prompting) compared to pretraining on the same amount of error-free data. We also delve into many details, such as (1) how this approach differs from beam search, (2) how such data can be prepared, (3) whether masking is needed on the erroneous tokens, (4) the amount of error required, (5) whether such data can be deferred to the fine-tuning stage, and many others.
     </details>

72. **IS TRANSFORMER A STOCHASTIC PARROT? A CASE STUDY IN SIMPLE ARITHMETIC TASK** [[pdf]](https://openreview.net/forum?id=tYVmxoRps3) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large pre-trained language models have demonstrated impressive capabilities, but  there is still much to learn about how they operate. In this study, we  conduct a investigation of the autoregressive transformer’s ability to  perform basic addition operations.   Specifically, by using causal analysis we found that a few different attention heads in the middle layers control the addition carry, with each head processing carries of different lengths. Due to the lack of globality in these attention heads, the model struggles to handle long-sequence addition tasks. By performing inference intervention on mistral-7B, partial task performance can be restored, with the accuracy on 20-digit long-sequence additions from 2\% to 38\%. Moreover, through fine-tuning, we discovered that the model still struggles to generalize carry chains beyond the training sequence length and the formation of the attention heads is crucial to the length generalization. Our research reveals how the model performs addition, and further provides insights into the debate on whether these models are merely statistical.
     </details>

73. **ImProver: Agent-Based Automated Proof Optimization** [[pdf]](https://arxiv.org/abs/2410.04753v1) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ImProver is presented, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean and is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been used to generate formal proofs of mathematical theorems in proofs assistants such as Lean. However, we often want to optimize a formal proof with respect to various criteria, depending on its downstream use. For example, we may want a proof to adhere to a certain style, or to be readable, concise, or modularly structured. Having suitably optimized proofs is also important for learning tasks, especially since human-written proofs may not optimal for that purpose. To this end, we study a new problem of automated proof optimization: rewriting a proof so that it is correct and optimizes for an arbitrary criterion, such as length or readability. As a first method for automated proof optimization, we present ImProver, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean. We find that naively applying LLMs to proof optimization falls short, and we incorporate various improvements into ImProver, such as the use of symbolic Lean context in a novel Chain-of-States technique, as well as error-correction and retrieval. We test ImProver on rewriting real-world undergraduate, competition, and research-level mathematics theorems, finding that ImProver is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
     </details>

74. **Improve Mathematical Reasoning in Language Models with Automated Process Supervision** [[pdf]](https://openreview.net/forum?id=KwPUQOQIKt) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Complex multi-step reasoning tasks, such as solving mathematical problems or generating code, remain a significant hurdle for even the most advanced large language models (LLMs). Verifying LLM outputs with an Outcome Reward Model (ORM) is a standard inference-time technique aimed at enhancing the reasoning performance of LLMs. However, this still proves insufficient for reasoning tasks with a lengthy or multi-hop reasoning chain, where the intermediate outcomes are neither properly rewarded nor penalized. Process supervision addresses this limitation by assigning intermediate rewards during the reasoning process. To date, the methods used to collect process supervision data have relied on either human annotation or per-step Monte Carlo estimation, both prohibitively expensive to scale, thus hindering the broad application of this technique. In response to this challenge, we propose a novel divide-and-conquer style Monte Carlo Tree Search (MCTS) algorithm named \textit{OmegaPRM} for the efficient collection of high-quality process supervision data. This algorithm swiftly identifies the first error in the Chain of Thought (CoT) with binary search and balances the positive and negative examples, thereby ensuring both efficiency and quality. As a result, we are able to collect over 1.5 million process supervision annotations to train Process Reward Models (PRMs). This fully automated process supervision alongside the weighted self-consistency algorithm is able to enhance LLMs' math reasoning performances. We improved the success rates of the instruction-tuned Gemini Pro model from 51\% to 69.4\% on MATH500 and from 86.4\% to 93.6\% on GSM8K. Similarly, we boosted the success rates of Gemma2 27B from 42.3\% to 58.2\% on MATH500 and from 74.0\% to 92.2\% on GSM8K. The entire process operates without any human intervention or supervision, making our method both financially and computationally cost-effective compared to existing methods.
     </details>

75. **Improving Complex Reasoning with Dynamic Prompt Corruption: A Soft Prompt Optimization Approach** [[pdf]](https://openreview.net/forum?id=h7Qz1ulnvF) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Prompt Tuning (PT) has emerged as a promising Parameter-Efficient Fine-Tuning (PEFT) approach by appending trainable continuous prompt vectors to the input, maintaining competitive performance with significantly fewer trainable parameters. While PT has shown effectiveness in enhancing task performance, particularly for classification tasks, its application to complex reasoning tasks has been largely overlooked. Our investigation reveals that PT provides limited improvement and may even degrade performance in reasoning tasks. This phenomenon suggests that soft prompts can positively impact certain instances while negatively affecting others, particularly during the latter stages of reasoning. To address these challenges, we propose a novel method called Dynamic Prompt Corruption (DPC), which seeks to optimize the use of soft prompts in reasoning tasks. DPC dynamically adjusts the influence of soft prompts based on their impact on the reasoning process. Specifically, it involves two key components: Dynamic Trigger and Dynamic Corruption. Dynamic Trigger measures the influence of soft prompts, determining whether their impact is beneficial or detrimental. Dynamic Corruption mitigates the negative effects of soft prompts by selectively masking key tokens that interfere with the reasoning process. We validate our approach through extensive experiments on various large language models (LLMs) and reasoning tasks, including GSM8K, MATH, and AQuA. The results demonstrate that Dynamic Prompt Corruption consistently improves the performance of LLMs, achieving  4\%-8\% accuracy gains compared to standard prompt tuning. These findings highlight the effectiveness of our approach and its potential to enhance complex reasoning in LLMs.
     </details>

76. **Improving LLM Reasoning through Scaling Inference Computation with Collaborative Verification** [[pdf]](http://arxiv.org/abs/2410.05318) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a comprehensive dataset consisting of correct and incorrect solutions for math and code tasks, generated by multiple LLMs, and proposes a novel collaborative method integrating Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in the general capability of large language models (LLMs), they continue to struggle with consistent and accurate reasoning, especially in complex tasks such as mathematical and code reasoning. One key limitation is that LLMs are trained primarily on correct solutions, reducing their ability to detect and learn from errors, which hampers their ability to reliably verify and rank outputs. To address this, we scale up the inference-time computation by generating multiple reasoning paths and employing verifiers to assess and rank the generated outputs by correctness. To facilitate this, we introduce a comprehensive dataset consisting of correct and incorrect solutions for math and code tasks, generated by multiple LLMs. This diverse set of solutions enables verifiers to more effectively distinguish and rank correct answers from erroneous outputs. The training methods for building verifiers were selected based on an extensive comparison of existing approaches. Moreover, to leverage the unique strengths of different reasoning strategies, we propose a novel collaborative method integrating Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification. CoT provides a clear, step-by-step reasoning process that enhances interpretability, while PoT, being executable, offers a precise and error-sensitive validation mechanism. By taking both of their strengths, our approach significantly improves the accuracy and reliability of reasoning verification. Our verifiers, Math-Rev and Code-Rev, demonstrate substantial performance gains to existing LLMs, achieving state-of-the-art results on benchmarks such as GSM8k and MATH and even outperforming GPT-4o with Qwen-72B-Instruct as the reasoner.
     </details>

77. **Improving Language Model Self-Correction Capability with Meta-Feedback** [[pdf]](https://openreview.net/forum?id=jJvJqgPZCD) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are capable of self-correcting their responses by generating feedback and refining the initial output. However, their performance may sometimes decline following self-correction, either because the feedback contains errors or due to unnecessarily attempting to refine an already accurate response. To address these limitations, we investigate whether the same LLM can generate meta-feedback that pinpoints errors in the feedback rather than the response, an ability that remains under-explored despite extensive research on LLMs' self-feedback generation. We design a novel self-correction prompting framework, Feedback-on-Feedback (FoF), which leverages meta-feedback to improve the feedback before refining the response. Our framework first samples multiple pieces of feedback for the initial response, and prompts the LLM to generate meta-feedback that analyzes the inconsistency between these feedback pieces. Based on the meta-feedback, the LLM generates refined feedback that subsequently guides the revision of the response. Our FoF framework consistently outperforms competitive baselines across two LLMs on three datasets, covering arithmetic reasoning, machine translation, and programming tasks. Specifically, FoF improves performance on GSM8K by 3.6 points (45.2% vs. 41.6% for the initial answer) and on MBPP by 6.4 points (51.7% vs. 45.3%) using the LLaMA-3-8B model.
     </details>

78. **Improving Reasoning Ability of Large Language Models via Iterative Uncertainty-based Preference Optimization** [[pdf]](https://openreview.net/forum?id=bGGMLWAGMc) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) has recently emerged as an efficient and effective method for aligning large language models with human preferences. However, constructing high-quality preference datasets remains challenging, often necessitating expensive manual or powerful LM annotations. Additionally, standard DPO exhibits suboptimal performance in complex reasoning tasks, such as mathematical and code reasoning. In this paper, we introduce an approach to collect preference pairs through iterative sampling and execution feedback, tailored to the current learning state (e.g. well-learned, mis-learned, and unlearned) of the policy model. To alleviate the failures of DPO and improve its applicability in reasoning tasks, we propose IUPO, an iterative uncertainty-based preference optimization method that achieves fine-grained preference control by assessing model confidence. We validate our approach across three reasoning tasks, incorporating five established reasoning datasets and one self-curated dataset. Our experimental results demonstrate an overall improvement of 3.6% over the standard DPO method.  Furthermore, our approach exhibits promising generalizability involving weak-to-strong (8B to 70B) and cross-model (Llama to Mistral) generalizations.
     </details>

79. **Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving** [[pdf]](https://openreview.net/forum?id=VNckp7JEHn) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While the scaling laws of large language models (LLMs) training have been extensively studied, optimal inference configurations of LLMs remain underexplored. We study _inference scaling laws_ and _compute-optimal inference_, focusing on the trade-offs between model sizes and generating additional tokens with different inference strategies. As a first step towards understanding and designing compute-optimal inference methods, we studied cost-performance trade-offs for inference strategies such as greedy search, majority voting, best-of-$n$, weighted voting, and two different tree search algorithms, using different model sizes and compute budgets. Our findings indicate smaller models (e.g., Llemma-7B) can outperform larger models given the same computation budgets, and that smaller models paired with advanced inference algorithms yield Pareto-optimal cost-performance trade-offs. For instance, the Llemma-7B model, equipped with our novel tree search algorithm, consistently outperforms Llemma-34B with standard majority voting on the MATH benchmark across all FLOPs budgets. We hope these findings contribute to a broader understanding of inference scaling laws for LLMs.
     </details>

80. **Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models** [[pdf]](https://openreview.net/forum?id=77gQUdQhE7) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent studies indicate that effectively utilizing inference-time compute is crucial for attaining good performance from large language models (LLMs). Specifically, the Best-of-N (BoN) inference strategy, where an LLM generates multiple responses and a verifier selects the best, has shown strong empirical performance. Motivated by this, we develop a novel inference-aware fine-tuning paradigm, which encompasses the BoN-aware inference framework as a special case. We devise the first imitation learning and reinforcement learning (RL) methods for fine-tuning LLMs using BoN, overcoming the challenging, non-differentiable argmax operator in BoN. We empirically demonstrate that our BoN-aware models implicitly learn a per-example "meta-strategy", which interleaves best responses with more diverse responses that might be better suited to a test-time input—a process reminiscent of the exploration-exploitation trade-off in RL. Our experiments demonstrate the effectiveness of BoN-aware fine-tuning in terms of improved performance and inference-time compute. In particular, we show that our methods improve the BoN performance of Gemma 2B on Hendrycks MATH from 26.8% to 30.8%, and Pass@K from 60% to 67%.
     </details>

81. **Interpretable Contrastive Monte Carlo Tree Search Reasoning** [[pdf]](http://arxiv.org/abs/2410.01707) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work conducted extensive ablation studies and quantitative analysis on components of MCTS, revealing the impact of each component on the MCTS reasoning performance of LLMs and designed a highly interpretable reward model based on the principle of contrastive decoding.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose SC-MCTS*: a novel Monte Carlo Tree Search (MCTS) reasoning algorithm for Large Language Models (LLMs), significantly improves both reasoning accuracy and speed. Our motivation comes from: 1. Previous MCTS LLM reasoning works often overlooked its biggest drawback--slower speed compared to CoT; 2. Previous research mainly used MCTS as a tool for LLM reasoning on various tasks with limited quantitative analysis or ablation studies of its components from reasoning interpretability perspective. 3. The reward model is the most crucial component in MCTS, however previous work has rarely conducted in-depth study or improvement of MCTS's reward models. Thus, we conducted extensive ablation studies and quantitative analysis on components of MCTS, revealing the impact of each component on the MCTS reasoning performance of LLMs. Building on this, (i) we designed a highly interpretable reward model based on the principle of contrastive decoding and (ii) achieved an average speed improvement of 51.9% per node using speculative decoding. Additionally, (iii) we improved UCT node selection strategy and backpropagation used in previous works, resulting in significant performance improvement. We outperformed o1-mini by an average of 17.4% on the Blocksworld multi-step reasoning dataset using Llama-3.1-70B with SC-MCTS*.
     </details>

82. **It Helps to Take a Second Opinion: Teaching Smaller LLMs To Deliberate Mutually via Selective Rationale Optimisation** [[pdf]](https://openreview.net/forum?id=NHxwxc3ql6) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Very large language models (LLMs) such as GPT-4 have shown the ability to handle complex tasks by generating and self-refining step-by-step rationales. Smaller language models (SLMs), typically with < 13B parameters, have been improved by using the data generated from very-large LMs through knowledge distillation. However, various practical constraints such as API costs, copyright, legal and ethical policies restrict using large (often opaque) models to train smaller models for commercial use. Limited success has been achieved at improving the ability of an SLM to explore the space of possible rationales and evaluate them by itself through self-deliberation. To address this, we propose COALITION, a trainable framework that facilitates interaction between two variants of the same SLM and trains them to generate and refine rationales optimized for the end-task. The variants exhibit different behaviors to produce a set of diverse candidate rationales during the generation and refinement steps. The model is then trained via Selective Rationale Optimization (SRO) to prefer generating rationale candidates that maximize the likelihood of producing the ground-truth answer. During inference, COALITION employs a controller to select the suitable variant for generating and refining the rationales. On five different datasets covering mathematical problems, commonsense reasoning, and natural language inference, COALITION outperforms several baselines by up to 5%. Our ablation studies reveal that cross-communication between the two variants performs better than using the single model to self-refine the rationales. We also demonstrate the applicability of COALITION for LMs of varying scales (4B to 14B parameters) and model families (Mistral, Llama, Qwen, Phi). We release the code for this work here.
     </details>

83. **LLM Spark: Critical Thinking Evaluation of Large Language Models** [[pdf]](https://openreview.net/forum?id=0sJ8TqOLGS) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) excel in complex tasks but often struggle with inconsistencies in problem framing, a critical skill for real-world scenarios. This paper introduces SPARK, a novel evaluation framework grounded in the Hierar- chical Three-Space Theory, to assess LLMs’ ability to identify missing informa- tion and challenge flawed problem setups. We create benchmarks by introducing inconsistencies and misleading cues in diverse question-answering datasets, cov- ering mathematics, science, and reading comprehension. Our experiments with state-of-the-art LLMs reveal their limitations in critical thinking, particularly in recognizing inconsistencies. We also explore mitigation strategies such as modi- fied prompting and targeted fine-tuning. Furthermore, we conduct comprehensive experiments to investigate how model and problem properties influence critical thinking capabilities in LLMs.
     </details>

84. **Language Model Non-Myopic Generation for Reasoning and Planning** [[pdf]](http://arxiv.org/abs/2410.17195) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models have demonstrated remarkable abilities in reasoning and planning by breaking down complex problems into sequential steps. Despite their success in various domains like mathematical problem-solving and coding, LLMs face challenges in ensuring reliable and optimal planning due to their inherent myopic nature of autoregressive decoding. This paper revisits LLM reasoning from an optimal-control perspective, proposing a novel method, Predictive-Decoding, that leverages Model Predictive Control to enhance planning accuracy. By re-weighting LLM distributions based on foresight trajectories, Predictive-Decoding aims to mitigate early errors and promote non-myopic planning. Our experiments show significant improvements in a wide range of tasks for math, coding, and agents. Furthermore, Predictive-Decoding demonstrates computational efficiency, outperforming search baselines with reduced computational resources. This study provides insights into optimizing LLM planning capabilities.
     </details>

85. **Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding** [[pdf]](https://openreview.net/forum?id=4Po8d9GAfQ) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown impressive capabilities, but still struggle with complex reasoning tasks requiring multiple steps. While prompt-based methods like Chain-of-Thought (CoT) can improve LLM reasoning at inference time, optimizing reasoning capabilities during training remains challenging. We introduce LaTent Reasoning Optimization (LaTRO), a principled framework that formulates reasoning as sampling from a latent distribution and optimizes it via variational approaches. LaTRO enables LLMs to concurrently improve both their reasoning process and ability to evaluate reasoning quality, without requiring external feedback or reward models. We validate LaTRO through experiments on GSM8K and ARC-Challenge datasets using multiple model architectures. On GSM8K, LaTRO improves zero-shot accuracy by an average of 12.5\% over base models and 9.6\% over supervised fine-tuning across Phi-3.5-mini, Mistral-7B, and Llama-3.1-8B. Our findings suggest that pre-trained LLMs possess latent reasoning capabilities that can be unlocked and enhanced through our proposed optimization approach in a self-improvement manner.
     </details>

86. **Language Models, Grade-School Math, and the Hidden Reasoning Process** [[pdf]](https://openreview.net/forum?id=Tn5B6Udq3E) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advances in language models have demonstrated their capability to solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K. In this paper, we formally study how language models solve these problems. We design a series of controlled experiments to address several fundamental questions: (1) Can language models truly develop reasoning skills, or do they simply memorize templates? (2) What is the model's hidden (mental) reasoning process? (3) Do models solve math questions using skills similar to or different from humans? (4) Do models trained on GSM8K-like datasets develop reasoning skills beyond those necessary for solving GSM8K problems? (5) What mental process causes models to make reasoning mistakes? (6) How large or deep must a model be to effectively solve GSM8K-level math questions?  Our study uncovers many hidden mechanisms by which language models solve mathematical questions, providing insights that extend beyond current understandings of LLMs.
     </details>

87. **Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models** [[pdf]](http://arxiv.org/abs/2410.01335) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.
     </details>


     <details>
          <summary>Abstract</summary>
          Model merging, such as model souping, is the practice of combining different models with the same architecture together without further training. In this work, we present a model merging methodology that addresses the difficulty of fine-tuning Large Language Models (LLMs) for target tasks in non-English languages, where task-specific data is often unavailable. We focus on mathematical reasoning and without in-language math data, facilitate cross-lingual transfer by composing language and math capabilities. Starting from the same pretrained model, we fine-tune separate "experts" on math instruction data in English and on generic instruction data in the target language. We then replace the top and bottom transformer layers of the math expert directly with layers from the language expert, which consequently enhances math performance in the target language. The resulting merged models outperform the individual experts and other merging methods on the math benchmark, MGSM, by 10% across four major languages where math instruction data is scarce. In addition, this layer swapping is simple, inexpensive, and intuitive, as it is based on an interpretative analysis of the most important parameter changes during the fine-tuning of each expert. The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.
     </details>

88. **Lean-ing on Quality: How High-Quality Data Beats Diverse Multilingual Data in AutoFormalization** [[pdf]](https://openreview.net/forum?id=Qdp7hlenr6) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization, the process of transforming informal mathematical language into formal specifications and proofs remains a difficult task for state-of-the-art (large) language models. Existing works point to competing explanations for the performance gap. On one hand, large language models exhibit exceptional performance on translation tasks, suggesting their significant potential for autoformalization.  On the other hand, the quantitative reasoning capabilities of standard language models remain limited, leading to suboptimal performance on autoformalization and the subsequent task of formal theorem proving. To this end, we introduce a novel methodology that leverages backtranslation with hand-curated prompts to enhance the mathematical capabilities of language models, particularly addressing the challenge posed by the scarcity of labeled data.  Specifically, we evaluate three primary variations of this strategy: (1) on-the-fly (online) backtranslation, (2) distilled (offline) backtranslation with few-shot amplification, and (3) line-by-line proof analysis integrated with proof state information. Each variant is designed to optimize data quality over quantity, focusing on the high fidelity of generated proofs rather than sheer data scale. Our findings provide evidence that employing our proposed approaches to generate synthetic data, which prioritizes quality over volume, improves the autoformalization performance of LLMs as measured by standard benchmarks such as ProofNet. Crucially, our approach outperforms pretrained models using a minimal number of tokens. We also show, through strategic prompting and backtranslation, that our approaches surpass the performance of finetuning with extensive multilingual datasets such as MMA on ProofNet with only 1/150th of the tokens. Taken together, our methods show a promising new approach to significantly reduce the resources required to formalize proofs, thereby accelerating AI for math.
     </details>

89. **LeanAgent: Lifelong Learning for Formal Theorem Proving** [[pdf]](http://arxiv.org/abs/2410.06209) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          LeanAgent is presented, a novel lifelong learning framework for theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge and achieves exceptional scores in stability and backward transfer.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have been successful in mathematical reasoning tasks such as formal theorem proving when integrated with interactive proof assistants like Lean. Existing approaches involve training or fine-tuning an LLM on a specific dataset to perform well on particular domains, such as undergraduate-level mathematics. These methods struggle with generalizability to advanced mathematics. A fundamental limitation is that these approaches operate on static domains, failing to capture how mathematicians often work across multiple domains and projects simultaneously or cyclically. We present LeanAgent, a novel lifelong learning framework for theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge. LeanAgent introduces several key innovations, including a curriculum learning strategy that optimizes the learning trajectory in terms of mathematical difficulty, a dynamic database for efficient management of evolving mathematical knowledge, and progressive training to balance stability and plasticity. LeanAgent successfully proves 162 theorems previously unproved by humans across 23 diverse Lean repositories, many from advanced mathematics. It performs significantly better than the static LLM baseline, proving challenging theorems in domains like abstract algebra and algebraic topology while showcasing a clear progression of learning from basic concepts to advanced topics. In addition, we analyze LeanAgent's superior performance on key lifelong learning metrics. LeanAgent achieves exceptional scores in stability and backward transfer, where learning new tasks improves performance on previously learned tasks. This emphasizes LeanAgent's continuous generalizability and improvement, explaining its superior theorem-proving performance.
     </details>

90. **Let's Be Self-generated via Step by Step: A Curriculum Learning Approach to Automated Reasoning with Large Language Models** [[pdf]](http://arxiv.org/abs/2410.21728) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel prompt approach for automatic reasoning named LBS3, inspired by curriculum learning which better reflects human learning habits is proposed, and it achieves strongly competitive performance compared to the SOTA baselines.
     </details>


     <details>
          <summary>Abstract</summary>
          While Chain of Thought (CoT) prompting approaches have significantly consolidated the reasoning capabilities of large language models (LLMs), they still face limitations that require extensive human effort or have performance needs to be improved. Existing endeavors have focused on bridging these gaps; however, these approaches either hinge on external data and cannot completely eliminate manual effort, or they fall short in effectively directing LLMs to generate high-quality exemplary prompts. To address the said pitfalls, we propose a novel prompt approach for automatic reasoning named \textbf{LBS3}, inspired by curriculum learning which better reflects human learning habits. Specifically, LBS3 initially steers LLMs to recall easy-to-hard proxy queries that are pertinent to the target query. Following this, it invokes a progressive strategy that utilizes exemplary prompts stemmed from easy-proxy queries to direct LLMs in solving hard-proxy queries, enabling the high-quality of the proxy solutions. Finally, our extensive experiments in various reasoning-intensive tasks with varying open- and closed-source LLMs show that LBS3 achieves strongly competitive performance compared to the SOTA baselines.
     </details>

91. **MAVIS: Mathematical Visual Instruction Tuning with an Automatic Data Engine** [[pdf]](https://openreview.net/forum?id=MnJzJ2gvuf) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models (MLLMs) have recently showcased superior proficiency in general visual scenarios. However, we identify their mathematical capabilities remain under-explored with three areas to be improved: visual encoding of math diagrams, diagram-language alignment, and chain-of-thought (CoT) reasoning. This draws forth an urgent demand for an effective training paradigm and a large-scale, comprehensive dataset with detailed CoT rationales, which is challenging to collect and costly to annotate manually. To tackle this issue, we propose MAVIS, a MAthematical VISual instruction tuning pipeline for MLLMs, featuring an automatic data engine to efficiently create mathematical visual datasets. We design the data generation process to be entirely independent of human intervention or GPT API usage, while ensuring the diagram-caption correspondence, question-answer correctness, and CoT reasoning quality. With this approach, we curate two datasets, MAVIS-Caption (558K diagram-caption pairs) and MAVIS-Instruct (834K visual math problems with CoT rationales), and propose four progressive stages for training MLLMs from scratch. First, we utilize MAVIS-Caption to fine-tune a math-specific vision encoder (CLIP-Math) through contrastive learning, tailored for improved diagram visual encoding. Second, we also leverage MAVIS-Caption to align the CLIP-Math with a large language model (LLM) by a projection layer, enhancing vision-language alignment in mathematical domains. Third, we adopt MAVIS-Instruct to perform the instruction tuning for robust problem-solving skills, and term the resulting model as MAVIS-7B. Fourth, we apply Direct Preference Optimization (DPO) to enhance the CoT capabilities of our model, further refining its step-wise reasoning performance. On various mathematical benchmarks, our MAVIS-7B achieves leading results among open-source MLLMs, e.g., surpassing other 7B models by +9.3% and the second-best LLaVA-NeXT (110B) by +6.9%, demonstrating the effectiveness of our method.
     </details>

92. **MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning** [[pdf]](https://arxiv.org/abs/2409.12147v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement, and employs a multi-agent loop with three agents.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models' (LLM) reasoning can be improved using test-time aggregation strategies, i.e., generating multiple samples and voting among generated samples. While these improve performance, they often reach a saturation point. Refinement offers an alternative by using LLM-generated feedback to improve solution quality. However, refinement introduces 3 key challenges: (1) Excessive refinement: Uniformly refining all instances can over-correct and reduce the overall performance. (2) Inability to localize and address errors: LLMs have a limited ability to self-correct and struggle to identify and correct their own mistakes. (3) Insufficient refinement: Deciding how many iterations of refinement are needed is non-trivial, and stopping too soon could leave errors unaddressed. To tackle these issues, we propose MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement. To improve error localization, we incorporate external step-wise reward model (RM) scores. Moreover, to ensure effective refinement, we employ a multi-agent loop with three agents: Solver, Reviewer (which generates targeted feedback based on step-wise RM scores), and the Refiner (which incorporates feedback). To ensure sufficient refinement, we re-evaluate updated solutions, iteratively initiating further rounds of refinement. We evaluate MAgICoRe on Llama-3-8B and GPT-3.5 and show its effectiveness across 5 math datasets. Even one iteration of MAgICoRe beats Self-Consistency by 3.4%, Best-of-k by 3.2%, and Self-Refine by 4.0% while using less than half the samples. Unlike iterative refinement with baselines, MAgICoRe continues to improve with more iterations. Finally, our ablations highlight the importance of MAgICoRe's RMs and multi-agent communication.
     </details>

93. **MIND: Math Informed syNthetic Dialogues for Pretraining LLMs** [[pdf]](http://arxiv.org/abs/2410.12881) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The utility of synthetic data to enhance pretraining data quality and hence to improve downstream task accuracy has been widely explored in recent large language models (LLMs). Yet, these approaches fall inadequate in complex, multi-hop and mathematical reasoning tasks as the synthetic data typically fails to add complementary knowledge to the existing raw corpus. In this work, we propose a novel large-scale and diverse Math Informed syNthetic Dialogue (MIND) generation method that improves the mathematical reasoning ability of LLMs. Specifically, using MIND, we generate synthetic conversations based on OpenWebMath (OWM), resulting in a new math corpus, MIND-OWM. Our experiments with different conversational settings reveal that incorporating knowledge gaps between dialog participants is essential for generating high-quality math data. We further identify an effective way to format and integrate synthetic and raw data during pretraining to maximize the gain in mathematical reasoning, emphasizing the need to restructure raw data rather than use it as-is. Compared to pretraining just on raw data, a model pretrained on MIND-OWM shows significant boost in mathematical reasoning (GSM8K: +13.42%, MATH: +2.30%), including superior performance in specialized knowledge (MMLU: +4.55%, MMLU-STEM: +4.28%) and general purpose reasoning tasks (GENERAL REASONING: +2.51%).
     </details>

94. **MIRAGE: Evaluating and Explaining Inductive Reasoning Process in Language Models** [[pdf]](http://arxiv.org/abs/2410.09542) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work evaluates LLMs' capabilities in both the inductive and deductive stages, allowing for flexible variation in input distribution, task scenario, and task difficulty to analyze the factors influencing LLMs' inductive reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Inductive reasoning is an essential capability for large language models (LLMs) to achieve higher intelligence, which requires the model to generalize rules from observed facts and then apply them to unseen examples. We present {\scshape Mirage}, a synthetic dataset that addresses the limitations of previous work, specifically the lack of comprehensive evaluation and flexible test data. In it, we evaluate LLMs' capabilities in both the inductive and deductive stages, allowing for flexible variation in input distribution, task scenario, and task difficulty to analyze the factors influencing LLMs' inductive reasoning. Based on these multi-faceted evaluations, we demonstrate that the LLM is a poor rule-based reasoner. In many cases, when conducting inductive reasoning, they do not rely on a correct rule to answer the unseen case. From the perspectives of different prompting methods, observation numbers, and task forms, models tend to consistently conduct correct deduction without correct inductive rules. Besides, we find that LLMs are good neighbor-based reasoners. In the inductive reasoning process, the model tends to focus on observed facts that are close to the current test example in feature space. By leveraging these similar examples, the model maintains strong inductive capabilities within a localized region, significantly improving its deductive performance.
     </details>

95. **Math for AI: On the Generalization of Learning Mathematical Problem Solving** [[pdf]](https://openreview.net/forum?id=th63j8qHa6) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          There has been a growing interest in enhancing the mathematical problem-solving (MPS) capabilities of LLMs. While some researchers focus on developing specialized math models to advance AI for math, others study mathematical reasoning with a math for AI perspective, positing that integrating mathematical reasoning data could enable LLMs to perform complex reasoning more broadly. This hypothesis draws from neuroscience studies which show that solving mathematical problems aids in the development of general reasoning skills in humans. The concept of ''math for AI'' has gained particular relevance as the research community increasingly focuses on complex reasoning -- Given the scarcity of complex and lengthy chain-of-thought data, MPS emerges as a prime candidate for collecting or synthesizing substantial volumes of intricate thought processes, thus serving as a potential key resource for enhancing general complex reasoning. However, it remains unclear whether skills acquired through learning MPS can extend to other reasoning tasks or merely improve MPS-specific benchmark scores.  In this paper, we present a comprehensive empirical analysis to address this question. Specifically, we explore three prevalent methods for improving MPS: (1) continual pretraining on mathematical text; (2) instruction pretraining on large-scale QA pairs synthesized from raw text; and (3) instruction tuning on MPS datasets.  Through controlled experiments and evaluations across seven distinct reasoning domains, we find that extensive continual pretraining on mathematical texts can improve performance on most non-MPS reasoning tasks generally. However, other dominant approaches of enhancing MPS performance fail to achieve significant gains on broad reasoning tasks. These findings indicate that most readily available data sources do not support the ''math for AI'' objective in enhancing non-MPS tasks. Identifying which data sources best contribute to the acquisition of complex reasoning skills remains a crucial question for future research.
     </details>

96. **MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code** [[pdf]](http://arxiv.org/abs/2410.08196) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method for generating mathematical code accompanied with corresponding reasoning steps for continued pretraining, leading to a 19.2B-token high-performing mathematical pretraining corpus, which is training several popular base models with this corpus significantly improves their mathematical abilities, leading to the creation of the MathCoder2 family of models.
     </details>


     <details>
          <summary>Abstract</summary>
          Code has been shown to be effective in enhancing the mathematical reasoning abilities of large language models due to its precision and accuracy. Previous works involving continued mathematical pretraining often include code that utilizes math-related packages, which are primarily designed for fields such as engineering, machine learning, signal processing, or module testing, rather than being directly focused on mathematical reasoning. In this paper, we introduce a novel method for generating mathematical code accompanied with corresponding reasoning steps for continued pretraining. Our approach begins with the construction of a high-quality mathematical continued pretraining dataset by incorporating math-related web data, code using mathematical packages, math textbooks, and synthetic data. Next, we construct reasoning steps by extracting LaTeX expressions, the conditions needed for the expressions, and the results of the expressions from the previously collected dataset. Based on this extracted information, we generate corresponding code to accurately capture the mathematical reasoning process. Appending the generated code to each reasoning step results in data consisting of paired natural language reasoning steps and their corresponding code. Combining this data with the original dataset results in a 19.2B-token high-performing mathematical pretraining corpus, which we name MathCode-Pile. Training several popular base models with this corpus significantly improves their mathematical abilities, leading to the creation of the MathCoder2 family of models. All of our data processing and training code is open-sourced, ensuring full transparency and easy reproducibility of the entire data collection and training pipeline. The code is released at https://github.com/mathllm/MathCoder2 .
     </details>

97. **MathEval: A Comprehensive Benchmark for Evaluating Large Language Models on Mathematical Reasoning Capabilities** [[pdf]](https://openreview.net/forum?id=DexGnh0EcB) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is a fundamental aspect of intelligence, encompassing a spectrum from basic arithmetic to intricate problem-solving. Recent investigations into the mathematical abilities of large language models (LLMs) have yielded inconsistent and incomplete assessments. In response, we introduce MathEval, a comprehensive benchmark designed to methodically evaluate the mathematical problem-solving proficiency of LLMs across varied contexts, adaptation strategies, and evaluation metrics. MathEval amalgamates 19 datasets, spanning an array of mathematical domains, languages, problem types, and difficulty levels, from elementary to advanced. This diverse collection facilitates a thorough appraisal of LLM performance and is stratified by language (English and Chinese), problem category (arithmetic, competitive mathematics, and higher mathematics), and difficulty. To overcome the challenges of standardizing responses across diverse models and prompts, we've developed an automated LLM-driven pipeline for answer extraction and comparison, ensuring consistent evaluation criteria. To broaden the utility of MathEval beyond the scope of GPT-4, we have harnessed the extensive results from GPT-4 to train a deepseek-7B-based answer comparison model, enabling precise answer validation for those without access to GPT-4. This model will also be made publicly available. MathEval not only assesses mathematical proficiency but also introduces a method to identify potential data contamination within pre-training datasets. This is done by hypothesizing that enhancements in one mathematical dataset should be mirrored by advancements in correlated datasets, thus signaling potential contamination—like the inadvertent inclusion of test data in the pre-training phase. To mitigate this and truly gauge progress, MathEval incorporates an annually refreshed set of problems from the latest Chinese National College Entrance Examination (Gaokao 2023), thereby benchmarking genuine advancements in mathematical problem solving skills. MathEval strives to refine the assessment of Large Language Models' (LLMs) capabilities in mathematics.
     </details>

98. **MathGAP: Out-of-Distribution Evaluation on Problems with Arbitrarily Complex Proofs** [[pdf]](https://openreview.net/forum?id=5ck9PIrTpH) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) can solve arithmetic word problems with high accuracy, but little is known about how well they generalize to problems that are more complex than the ones on which they have been trained. Empirical investigations of such questions are impeded by two major flaws of current evaluations: (i) much of the evaluation data is contaminated, in the sense that it has already been seen during training, and (ii) benchmark datasets do not capture how problem proofs may be arbitrarily complex in various ways. As a step towards addressing these issues, we present a framework for evaluating LLMs on problems with arbitrarily complex arithmetic proofs, called MathGAP. MathGAP generates problems that follow fixed proof specifications -- along with chain-of-thought reasoning annotations -- enabling systematic studies on generalization with respect to arithmetic proof complexity. We apply MathGAP to analyze how in-context learning interacts with generalization to problems that have more complex proofs. We find that among the models tested, most show a significant decrease in performance as proofs get deeper and wider. This effect is more pronounced in complex, nonlinear proof structures, which are challenging even for GPT-4o. Surprisingly, providing in-context examples from the same distribution as the test set is not always beneficial for performance. In particular, zero-shot prompting as well as demonstrating a diverse range of examples that are less complex than the test data sometimes yield similar or higher accuracies.
     </details>

99. **MathGLM-Vision: Solving Mathematical Problems with Multi-Modal Large Language Model** [[pdf]](http://arxiv.org/abs/2409.13729) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated significant capabilities in mathematical reasoning, particularly with text-based mathematical problems. However, current multi-modal large language models (MLLMs), especially those specialized in mathematics, tend to focus predominantly on solving geometric problems but ignore the diversity of visual information available in other areas of mathematics. Moreover, the geometric information for these specialized mathematical MLLMs is derived from several public datasets, which are typically limited in diversity and complexity. To address these limitations, we aim to construct a fine-tuning dataset named MathVL, and develop a series of specialized mathematical MLLMs termed MathGLM-Vision by conducting Supervised Fine-Tuning (SFT) on MathVL with various parameter-scale backbones. To extensively evaluate the effectiveness of MathGLM-Vision, we conduct experiments on several public benchmarks and our curated MathVL-test consisting of 2,000 problems. Experimental results demonstrate that MathGLM-Vision achieves significant improvements compared with some existing models, including backbone models and open-source mathematical MLLMs. These findings indicate the importance of diversity dataset in enhancing the mathematical reasoning abilities of MLLMs.
     </details>

100. **MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs** [[pdf]](http://arxiv.org/abs/2410.04698) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces MathHay, an automated benchmark designed to assess the long-context mathematical reasoning capabilities of LLMs, and conducts extensive experiments on MathHay to assess the long-context mathematical reasoning abilities of eight top-performing LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent large language models (LLMs) have demonstrated versatile capabilities in long-context scenarios. Although some recent benchmarks have been developed to evaluate the long-context capabilities of LLMs, there is a lack of benchmarks evaluating the mathematical reasoning abilities of LLMs over long contexts, which is crucial for LLMs' application in real-world scenarios. In this paper, we introduce MathHay, an automated benchmark designed to assess the long-context mathematical reasoning capabilities of LLMs. Unlike previous benchmarks like Needle in a Haystack, which focus primarily on information retrieval within long texts, MathHay demands models with both information-seeking and complex mathematical reasoning abilities. We conduct extensive experiments on MathHay to assess the long-context mathematical reasoning abilities of eight top-performing LLMs. Even the best-performing model, Gemini-1.5-Pro-002, still struggles with mathematical reasoning over long contexts, achieving only 51.26% accuracy at 128K tokens. This highlights the significant room for improvement on the MathHay benchmark.
     </details>

101. **MathScape: Evaluating MLLMs in Multi-modal Math Scenarios through a Hierarchical Benchmark** [[pdf]](https://openreview.net/forum?id=3jvgm61l9S) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          With the development of Multimodal Large Language Models (MLLMs), the evaluation of multimodal models in the context of mathematical problems has become a valuable research field. Multimodal visual-textual mathematical reasoning serves as a critical  indicator for evaluating the comprehension and complex multi-step quantitative reasoning abilities of MLLMs. However, previous multimodal math benchmarks have not sufficiently integrated visual and textual information. To address this gap, we proposed MathScape, a new benchmark that emphasizes the understanding and application of combined visual and textual information. MathScape is designed to evaluate photo-based math problem scenarios, assessing the theoretical understanding and application ability of MLLMs through a categorical hierarchical approach. We conduct a multi-dimensional evaluation on 11 advanced MLLMs, revealing that our benchmark is challenging even for the most sophisticated models. By analyzing the evaluation results, we identify the limitations of MLLMs, offering valuable insights for enhancing model performance.
     </details>

102. **Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solver** [[pdf]](https://openreview.net/forum?id=6aHUmotXaw) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper introduces rStar, a self-play mutual reasoning approach that significantly improves reasoning capabilities of small language models (SLMs) without fine-tuning or superior models. rStar decouples reasoning into a self-play mutual generation-discrimination process. First, a target SLM augments the Monte Carlo Tree Search (MCTS) with a rich set of human-like reasoning actions to construct higher quality reasoning trajectories. Next, another SLM, with capabilities similar to the target SLM, acts as a discriminator to verify each trajectory generated by the target SLM. The mutually agreed reasoning trajectories are considered mutual consistent, thus are more likely to be correct. Extensive experiments across five SLMs demonstrate rStar can effectively solve diverse reasoning problems, including GSM8K, GSM-Hard, MATH, SVAMP, and StrategyQA. Remarkably, rStar boosts GSM8K accuracy from 12.51\% to 63.91\% for LLaMA2-7B, from 36.46\% to 81.88\% for Mistral-7B, from 74.53\% to 91.13\% for LLaMA3-8B-Instruct.
     </details>

103. **Number Cookbook: Number Understanding of Language Models and How to Improve It** [[pdf]](https://openreview.net/forum?id=BWS5gVjgeY) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) can solve an increasing number of complex reasoning tasks while making surprising mistakes in basic numerical understanding and processing (such as 9.11 > 9.9). The latter ability is essential for tackling complex arithmetic and mathematical problems and serves as a foundation for most reasoning tasks, but previous work paid little attention to it or only discussed several restricted tasks (like integer addition). In this paper, we comprehensively investigate the numerical understanding and processing ability (NUPA) of LLMs. Firstly, we introduce a benchmark covering four common numerical representations and 17 distinct numerical tasks in four major categories, resulting in 41 meaningful combinations in total. These tasks are derived from primary and secondary education curricula, encompassing nearly all everyday numerical understanding and processing scenarios, and the rules of these tasks are very simple and clear. Through the benchmark, we find that current LLMs fail frequently in many of the tasks. To study the problem, we train small models with existing and potential techniques for enhancing NUPA (such as special tokenizers, PEs, and number formats), comprehensively evaluating their effectiveness using our testbed. We also finetune practical-scale LLMs on our proposed NUPA tasks and find that 1) naive finetuning can improve NUPA a lot on many but not all tasks, and 2) surprisingly, techniques designed to enhance NUPA prove ineffective for finetuning pretrained models. We further explore the impact of chain-of-thought techniques on NUPA. Our work takes a preliminary step towards understanding and improving NUPA of LLMs. Our benchmark and code are released at https://github.com/GraphPKU/number_cookbook.
     </details>

104. **Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models** [[pdf]](http://arxiv.org/abs/2410.07985) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a comprehensive and challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level, and shows that even the most advanced models, OpenAI o1-mini and OpenAI o1-preview, struggle with highly challenging Olympiad-level problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models (LLMs) have led to significant breakthroughs in mathematical reasoning capabilities. However, existing benchmarks like GSM8K or MATH are now being solved with high accuracy (e.g., OpenAI o1 achieves 94.8% on MATH dataset), indicating their inadequacy for truly challenging these models. To bridge this gap, we propose a comprehensive and challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level. Unlike existing Olympiad-related benchmarks, our dataset focuses exclusively on mathematics and comprises a vast collection of 4428 competition-level problems with rigorous human annotation. These problems are meticulously categorized into over 33 sub-domains and span more than 10 distinct difficulty levels, enabling a holistic assessment of model performance in Olympiad-mathematical reasoning. Furthermore, we conducted an in-depth analysis based on this benchmark. Our experimental results show that even the most advanced models, OpenAI o1-mini and OpenAI o1-preview, struggle with highly challenging Olympiad-level problems, with 60.54% and 52.55% accuracy, highlighting significant challenges in Olympiad-level mathematical reasoning.
     </details>

105. **On Designing Effective RL Reward at Training Time for LLM Reasoning** [[pdf]](http://arxiv.org/abs/2410.15115) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Reward models have been increasingly critical for improving the reasoning capability of LLMs. Existing research has shown that a well-trained reward model can substantially improve model performances at inference time via search. However, the potential of reward models during RL training time still remains largely under-explored. It is currently unclear whether these reward models can provide additional training signals to enhance the reasoning capabilities of LLMs in RL training that uses sparse success rewards, which verify the correctness of solutions. In this work, we evaluate popular reward models for RL training, including the Outcome-supervised Reward Model (ORM) and the Process-supervised Reward Model (PRM), and train a collection of LLMs for math problems using RL by combining these learned rewards with success rewards. Surprisingly, even though these learned reward models have strong inference-time performances, they may NOT help or even hurt RL training, producing worse performances than LLMs trained with the success reward only. Our analysis reveals that an LLM can receive high rewards from some of these reward models by repeating correct but unnecessary reasoning steps, leading to a severe reward hacking issue. Therefore, we introduce two novel reward refinement techniques, including Clipping and Delta. The key idea is to ensure the accumulative reward of any reasoning trajectory is upper-bounded to keep a learned reward model effective without being exploited. We evaluate our techniques with multiple reward models over a set of 1.5B and 7B LLMs on MATH and GSM8K benchmarks and demonstrate that with a carefully designed reward function, RL training without any additional supervised tuning can improve all the evaluated LLMs, including the state-of-the-art 7B LLM Qwen2.5-Math-7B-Instruct on MATH and GSM8K benchmarks.
     </details>

106. **Open Eyes, Then Reason: Fine-grained Visual Mathematical Understanding in MLLMs** [[pdf]](https://openreview.net/forum?id=6zAIFLgayn) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Current multimodal large language models (MLLMs) often underperform on mathematical problem-solving tasks that require fine-grained visual understanding. The limitation primarily arises from inadequate perception of geometric primitives during image-level contrastive pre-training (e.g., CLIP). Current efforts to enhance MLLM performance have focused on scaling up mathematical visual instruction datasets and employing stronger LLM backbones, yet these approaches often neglect persistent visual recognition errors in MLLMs. In this paper, we systematically evaluate the visual grounding capabilities of state-of-the-art MLLMs and uncover a negative correlation between their visual grounding accuracy and problem-solving performance. Notably, even advanced models like GPT-4o demonstrate a significant error rate (70\%) when identifying geometric entities, highlighting that fine-grained visual understanding remains a crucial bottleneck in visual mathematical reasoning. To address this, we propose a novel approach, SVE-Math (Selective Vision-Enhanced Mathematical MLLM), featuring a geometric-grounded vision encoder and a feature router that dynamically adjusts the contribution of hierarchical visual feature maps. Our model recognizes accurate visual primitives and generates precise visual prompts tailored to the language model's reasoning needs. In experiments, SVE-Math-7B outperforms other 7B-parameter models by 5.5\% on MathVerse and even surpasses larger models on MathVista. Despite being trained on smaller datasets, SVE-Math-7B matches the performance of models trained on significantly larger datasets on GeoQA benchmark. Our findings provide critical insights for future research, highlighting the need for more effective integration of fine-grained visual understanding in MLLMs.  We will release model weights, code, and instructions upon acceptance.
     </details>

107. **Optimizing Inference-Time Reasoning in LLMs via Retrieval-Augmented Reflection** [[pdf]](https://openreview.net/forum?id=ElYRG3pJcv) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Empowering LLMs to improve their performance through increased inference-time computation is a crucial step in developing self-improving agents capable of operating in open-ended natural language contexts. In this paper, we explore how iterative revising a chain of thoughts with the help of information retrieval significantly improves large language models' reasoning ability in challenging tasks, while hugely mitigating hallucination. In particular, the proposed method --- \emph{retrieval-augmented reflection} (RaR) --- revises the generation tokens step one by one with multiple retrieved information relevant to the instruction. Applying RaR during inference-time to a various set of language models substantially improves their performances on various reasoning tasks; on relatively increasing scores by 13.63\% on code generation, 16.96\% on mathematical reasoning, and 42.78\% on embodied task planning. Moreover, we find that with more inference-time computation given to the LLM for multi-times retrieval-augmented reflection, the LLM can continuously improve on various reasoning benchmarks. With lower inference-time computation (FLOPs), a small LM can surpass the performance of the LM with more than 10 times the parameters.
     </details>

108. **Paramanu-Ganita: An Efficient Pre-trained Generative Mathematics Language Model with Chain-of-Thought Instruction Fine-Tuning** [[pdf]](https://openreview.net/forum?id=v3DwQlyGbv) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we present PARAMANU -G ANITA, a 208 million-parameter novel Auto Regressive (AR) decoder based language model on mathematics. We per- formed pretraining from scratch on 31.5 billion tokens using a context size of 4096 on a mixed mathematical corpus consisting of mathematical web pages, mathe- matics related source code such as AlgebraStack, mathematical textbooks, Chain- of-Thought (CoT) templatised mathematical StackOverflow question answers pairs, and mathematical lecture notes in LATEX curated by us. We also trained a math and code specialised BPE tokenizer. We proposed and performed Chain-of- Thought instruction fine-tuning of Paramanu-Ganita on the MetaMathQA dataset. We evaluate our model on GSM8K and MATH mathematical benchmarks, and on logical deductive reasoning (LogiQA) and multiple choice high school and col- lege level math questions from SAT (AGIEVAL-SAT-Math), GRE/GMAT ques- tions (AGIEVAL-AQuA-RAT), college and high school level math questions from MMLU. Our model Paramanu-Ganita, despite being 34 times smaller than the 7B LLMs, outperforms general LLMs by approximately 30% points, and even math-specialised LLMs by 3-23% points in GSM8K test accuracy metric. On MATH benchmark, Paramanu-Ganita outperformed the various models by 6-8% points. On other benchmarks such as LogiQA logical deductive reasoning bench- mark, mathematical high school level multi-choice questions (MMLU-math-high- school), GRE-GMAT level quantitative questions (AGIEVAL-AQuA-RAT), SAT level math questions, Paramanu-Ganita was better than the others by about 1-4% points. The large significant margin improvement in performance of our math model over the existing LLMs signifies that reasoning capabilities of language models are just not restricted to those with humongous number of parameters. Paramanu-Ganita took only 170 hours of A100 training whereas large LLMs such as the math-specialised LLM, LLEMMA 7B, was trained for 23,000 A100 equiv- alent hours. Thus, our approach of pretraining powerful domain-specialised lan- guage models from scratch for domain adaptation is much more cost-effective and environmental friendly than performing continual training of LLMs.
     </details>

109. **PersonaMath: Enhancing Math Reasoning through Persona-Driven Data Augmentation** [[pdf]](http://arxiv.org/abs/2410.01504) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a data augmentation approach and introduces PersonaMathQA, a dataset derived from MATH and GSM8K, on which the PersonaMath models are trained, and introduces a novel persona-driven data augmentation technique to enhance the dataset's quantity and diversity.
     </details>


     <details>
          <summary>Abstract</summary>
          While closed-source Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities, open-source models continue to struggle with such tasks. To bridge this gap, we propose a data augmentation approach and introduce PersonaMathQA, a dataset derived from MATH and GSM8K, on which we train the PersonaMath models. Our approach consists of two stages: the first stage is learning from Persona Diversification, and the second stage is learning from Reflection. In the first stage, we regenerate detailed chain-of-thought (CoT) solutions as instructions using a closed-source LLM and introduce a novel persona-driven data augmentation technique to enhance the dataset's quantity and diversity. In the second stage, we incorporate reflection to fully leverage more challenging and valuable questions. Evaluation of our PersonaMath models on MATH and GSM8K reveals that the PersonaMath-7B model (based on LLaMA-2-7B) achieves an accuracy of 24.2% on MATH and 68.7% on GSM8K, surpassing all baseline methods and achieving state-of-the-art performance. Notably, our dataset contains only 70.3K data points-merely 17.8% of MetaMathQA and 27% of MathInstruct-yet our model outperforms these baselines, demonstrating the high quality and diversity of our dataset, which enables more efficient model training. We open-source the PersonaMathQA dataset, PersonaMath models, and our code for public usage.
     </details>

110. **Polymath: A Challenging Multi-modal Mathematical Reasoning Benchmark** [[pdf]](http://arxiv.org/abs/2410.14702) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models (MLLMs) exhibit impressive problem-solving abilities in various domains, but their visual comprehension and abstract reasoning skills remain under-evaluated. To this end, we present PolyMATH, a challenging benchmark aimed at evaluating the general cognitive reasoning abilities of MLLMs. PolyMATH comprises 5,000 manually collected high-quality images of cognitive textual and visual challenges across 10 distinct categories, including pattern recognition, spatial reasoning, and relative reasoning. We conducted a comprehensive, and quantitative evaluation of 15 MLLMs using four diverse prompting strategies, including Chain-of-Thought and Step-Back. The best scores achieved on PolyMATH are ~41%, ~36%, and ~27%, obtained by Claude-3.5 Sonnet, GPT-4o and Gemini-1.5 Pro respectively - highlighting the logical and visual complexity of these questions. A further fine-grained error analysis reveals that these models struggle to understand spatial relations and perform drawn-out, high-level reasoning. This is further strengthened by our ablation study estimating MLLM performance when given textual descriptions in place of diagrams. As evidenced by ~4% improvement over textual descriptions as opposed to actual images, we discover that models do not truly comprehend visual diagrams and the spatial information therein, and are thus prone to logical errors. Finally, we evaluate the OpenAI o1 models and find that their performance only matches the human baseline, highlighting the difficulty of the benchmark. The results on PolyMATH highlight the room for improvement in multi-modal reasoning and provide unique insights to guide the development of future MLLMs.
     </details>

111. **Preference Optimization for Reasoning with Pseudo Feedback** [[pdf]](https://openreview.net/forum?id=jkUp3lybXf) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Preference optimization techniques, such as Direct Preference Optimization (DPO), are frequently employed to enhance the reasoning capabilities of large language models (LLMs) in domains like mathematical reasoning and coding, typically following supervised fine-tuning. These methods rely on high-quality labels for reasoning tasks to generate preference pairs; however, the availability of reasoning datasets with human-verified labels is limited. In this study, we introduce a novel approach to generate pseudo feedback for reasoning tasks by framing the labeling of solutions to reason problems as an evaluation against associated \emph{test cases}.  We explore two forms of pseudo feedback based on test cases: one generated by frontier LLMs and the other by extending self-consistency to multi-test-case. We conduct experiments on both mathematical reasoning and coding tasks using pseudo feedback for preference optimization, and observe improvements across both tasks. Specifically, using Mathstral-7B as our base model, we improve MATH results from 58.3 to 68.6, surpassing both NuminaMath-72B and GPT-4-Turbo-1106-preview. In GSM8K and College Math, our scores increase from 85.6 to 90.3 and from 34.3 to 42.3, respectively. Building on Deepseek-coder-7B-v1.5, we achieve a score of 24.3 on LiveCodeBench (from 21.1), surpassing Claude-3-Haiku.
     </details>

112. **Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning** [[pdf]](https://openreview.net/forum?id=FiyS0ecSm0) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~\ref{fig:example}). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
     </details>

113. **Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs** [[pdf]](https://openreview.net/forum?id=WrBqgoseGL) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          As large language models (LLMs) continue to advance, many existing benchmarks designed to evaluate their reasoning capabilities are becoming saturated. Therefore, we present the Putnam-AXIOM Original benchmark consisting of 236 mathematical problems from the William Lowell Putnam Mathematical Competition, along with detailed step-by-step solutions. To preserve the Putnam-AXIOM benchmark's validity and mitigate potential data contamination, we created the Putnam-AXIOM Variation benchmark with functional variations of 52 problems. By programmatically altering problem elements like variables and constants, we can generate unlimited novel, equally challenging problems not found online. We see that almost all models have significantly lower accuracy in the variations than the original problems. Our results reveal that OpenAI's o1-preview, the best performing model, achieves merely 41.95\% accuracy on the Putnam-AXIOM Original but experiences around a 30\% reduction in accuracy on the variations' dataset when compared to corresponding original problems. Moreover, we explore metrics beyond boxed accuracy to assess models on complex tasks like natural language theorem proving, crucial for evaluating reasoning capabilities  in depth, opening the possibility for open-ended evaluation of reasoning strings.
     </details>

114. **R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models** [[pdf]](http://arxiv.org/abs/2410.17885) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Existing Large Multimodal Models (LMMs) struggle with mathematical geometric reasoning due to a lack of high-quality image-text paired data. Current geometric data generation approaches, which apply preset templates to generate geometric data or use Large Language Models (LLMs) to rephrase questions and answers (Q&A), unavoidably limit data accuracy and diversity. To synthesize higher-quality data, we propose a two-stage Reverse Chain-of-Thought (R-CoT) geometry problem generation pipeline. First, we introduce GeoChain to produce high-fidelity geometric images and corresponding descriptions highlighting relations among geometric elements. We then design a Reverse A&Q method that reasons step-by-step based on the descriptions and generates questions in reverse from the reasoning results. Experiments demonstrate that the proposed method brings significant and consistent improvements on multiple LMM baselines, achieving new performance records in the 2B, 7B, and 8B settings. Notably, R-CoT-8B significantly outperforms previous state-of-the-art open-source mathematical models by 16.6% on MathVista and 9.2% on GeoQA, while also surpassing the closed-source model GPT-4o by an average of 13% across both datasets. The code is available at https://github.com/dle666/R-CoT.
     </details>

115. **RLSF: Reinforcement Learning from Self-feedback for improved logical reasoning** [[pdf]](https://openreview.net/forum?id=gdzpnRBP4F) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated impressive capabilities in generating coherent and contextually relevant text. These models arguably lack the ability to logically reason, an essential skill required to solving mathematical problems and programming tasks. While step-by-step prompting approaches show some promise, they often depend on finding a suitable prompt tailored to the specific model and task.  In this work, we propose a simple, yet an effective approach to enhance reasoning capabilities by leveraging reinforcement learning (RL) and the confidence scores of a well-calibrated LLM. It involves optimising an implicit reward derived from the model's confidence levels in the answer to the reasoning task at hand. We generate preference data and fine-tune the LLM in a similar spirit to reinforcement learning from human feedback (RLHF), but without needing any human provided labels or preferences. Our results show that resulting reasoning abilities of an LLM improve and are transferable to other reasoning tasks. This warrants further investigation of RL as a facilitator for solving complex language tasks.
     </details>

116. **ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement** [[pdf]](http://arxiv.org/abs/2410.02108) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods.
     </details>


     <details>
          <summary>Abstract</summary>
          Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.
     </details>

117. **Reasoning Elicitation in Language Models via Counterfactual Feedback** [[pdf]](http://arxiv.org/abs/2410.03767) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work derives novel metrics that balance accuracy in factual and counterfactual questions, capturing a more complete view of the reasoning abilities of language models than traditional factual-only based metrics.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the increasing effectiveness of language models, their reasoning capabilities remain underdeveloped. In particular, causal reasoning through counterfactual question answering is lacking. This work aims to bridge this gap. We first derive novel metrics that balance accuracy in factual and counterfactual questions, capturing a more complete view of the reasoning abilities of language models than traditional factual-only based metrics. Second, we propose several fine-tuning approaches that aim to elicit better reasoning mechanisms, in the sense of the proposed metrics. Finally, we evaluate the performance of the fine-tuned language models in a variety of realistic scenarios. In particular, we investigate to what extent our fine-tuning approaches systemically achieve better generalization with respect to the base models in several problems that require, among others, inductive and deductive reasoning capabilities.
     </details>

118. **Rethinking and improving autoformalization: towards a faithful metric and a Dependency Retrieval-based approach** [[pdf]](https://openreview.net/forum?id=hUb2At2DsQ) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          As a central component in formal verification, statement autoformalization has been widely studied including the recent efforts from machine learning community, but still remains a widely-recognized difficult and open problem. In this paper, we delve into two critical yet under-explored gaps: 1) absence of faithful and universal automated evaluation for autoformalization results; 2) agnosia of contextural information, inducing severe hallucination of formal definitions and theorems. To address the first issue, we propose **BEq** (_**B**idirectional **E**xtended Definitional E**q**uivalence_), an automated neuro-symbolic method to determine the equivalence between two formal statements, which is formal-grounded and well-aligned with human intuition. For the second, we propose **RAutoformalizer** (_**R**etrieval-augmented **Autoformalizer**_), augmenting statement autoformalization by _Dependency Retrieval_, retrieving potentially dependent objects from formal libraries. We parse the dependencies of libraries and propose to _structurally informalise_ formal objects by the topological order of dependencies. To evaluate OOD generalization and research-level capabilities, we build a novel benchmark, _Con-NF_, consisting of 961 informal-formal statement pairs from frontier mathematical researches. Extensive experiments validate the effectiveness of our proposed approaches. In particular, BEq is evaluated on 200 diverse formal statement pairs with expert-annotated equivalence label, exhibiting significantly improved accuracy ($82.50\\% \mapsto 90.50\\%$) and precision ($70.59\\% \mapsto 100.0\\%$). For dependency retrieval, a baseline with excellent performance is established. The proposed RAutoformalizer substantially outperforms SOTA baselines in both in-distribution ProofNet benchmark ($12.83\\% \mapsto 18.18\\%$, BEq@8) and OOD Con-NF scenario ($4.58\\%\mapsto 16.86\\%$, BEq@8). Code, data, and models will be available.
     </details>

119. **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** [[pdf]](http://arxiv.org/abs/2410.08146) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work describes the set of good provers and shows that optimizing process rewards from such provers improves exploration during test-time search and online RL and shows that weak prover policies can substantially improve a stronger base policy.
     </details>


     <details>
          <summary>Abstract</summary>
          A promising approach for improving reasoning in large language models is to use process reward models (PRMs). PRMs provide feedback at each step of a multi-step reasoning trace, potentially improving credit assignment over outcome reward models (ORMs) that only provide feedback at the final step. However, collecting dense, per-step human labels is not scalable, and training PRMs from automatically-labeled data has thus far led to limited gains. To improve a base policy by running search against a PRM or using it as dense rewards for reinforcement learning (RL), we ask: "How should we design process rewards?". Our key insight is that, to be effective, the process reward for a step should measure progress: a change in the likelihood of producing a correct response in the future, before and after taking the step, corresponding to the notion of step-level advantages in RL. Crucially, this progress should be measured under a prover policy distinct from the base policy. We theoretically characterize the set of good provers and our results show that optimizing process rewards from such provers improves exploration during test-time search and online RL. In fact, our characterization shows that weak prover policies can substantially improve a stronger base policy, which we also observe empirically. We validate our claims by training process advantage verifiers (PAVs) to predict progress under such provers, and show that compared to ORMs, test-time search against PAVs is $>8\%$ more accurate, and $1.5-5\times$ more compute-efficient. Online RL with dense rewards from PAVs enables one of the first results with $5-6\times$ gain in sample efficiency, and $>6\%$ gain in accuracy, over ORMs.
     </details>

120. **SIKeD: Self-guided Iterative Knowledge Distillation for mathematical reasoning** [[pdf]](http://arxiv.org/abs/2410.18574) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) can transfer their reasoning skills to smaller models by teaching them to generate the intermediate reasoning process required to solve multistep reasoning tasks. While LLMs can accurately solve reasoning tasks through a variety of strategies, even without fine-tuning, smaller models are not expressive enough to fit the LLMs distribution on all strategies when distilled and tend to prioritize one strategy over the others. This reliance on one strategy poses a challenge for smaller models when attempting to solve reasoning tasks that may be difficult with their preferred strategy. To address this, we propose a distillation method SIKeD (Self-guided Iterative Knowledge Distillation for mathematical reasoning), where the LLM teaches the smaller model to approach a task using different strategies and the smaller model uses its self-generated on-policy outputs to choose the most suitable strategy for the given task. The training continues in a self-guided iterative manner, where for each training iteration, a decision is made on how to combine the LLM data with the self-generated outputs. Unlike traditional distillation methods, SIKeD allows the smaller model to learn which strategy is suitable for a given task while continuously learning to solve a task using different strategies. Our experiments on various mathematical reasoning datasets show that SIKeD significantly outperforms traditional distillation techniques across smaller models of different sizes. Our code is available at: https://github.com/kumar-shridhar/SIKeD
     </details>

121. **SIaM: Self-Improving Code-Assisted Mathematical Reasoning of Large Language Models** [[pdf]](https://arxiv.org/abs/2408.15565v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel paradigm that uses a code-based critic model to guide steps including question-code data construction, quality control, and complementary evaluation and explores different alignment algorithms with self-generated instruction/preference data to foster continuous improvement.
     </details>


     <details>
          <summary>Abstract</summary>
          There is a growing trend of teaching large language models (LLMs) to solve mathematical problems through coding. Existing studies primarily focus on prompting powerful, closed-source models to generate seed training data followed by in-domain data augmentation, equipping LLMs with considerable capabilities for code-aided mathematical reasoning. However, continually training these models on augmented data derived from a few datasets such as GSM8K may impair their generalization abilities and restrict their effectiveness to a narrow range of question types. Conversely, the potential of improving such LLMs by leveraging large-scale, expert-written, diverse math question-answer pairs remains unexplored. To utilize these resources and tackle unique challenges such as code response assessment, we propose a novel paradigm that uses a code-based critic model to guide steps including question-code data construction, quality control, and complementary evaluation. We also explore different alignment algorithms with self-generated instruction/preference data to foster continuous improvement. Experiments across both in-domain (up to +5.7%) and out-of-domain (+4.4%) benchmarks in English and Chinese demonstrate the effectiveness of the proposed paradigm.
     </details>

122. **SMART: Self-learning Meta-strategy Agent for Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2410.16128) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Tasks requiring deductive reasoning, especially those involving multiple steps, often demand adaptive strategies such as intermediate generation of rationales or programs, as no single approach is universally optimal. While Language Models (LMs) can enhance their outputs through iterative self-refinement and strategy adjustments, they frequently fail to apply the most effective strategy in their first attempt. This inefficiency raises the question: Can LMs learn to select the optimal strategy in the first attempt, without a need for refinement? To address this challenge, we introduce SMART (Self-learning Meta-strategy Agent for Reasoning Tasks), a novel framework that enables LMs to autonomously learn and select the most effective strategies for various reasoning tasks. We model the strategy selection process as a Markov Decision Process and leverage reinforcement learning-driven continuous self-improvement to allow the model to find the suitable strategy to solve a given task. Unlike traditional self-refinement methods that rely on multiple inference passes or external feedback, SMART allows an LM to internalize the outcomes of its own reasoning processes and adjust its strategy accordingly, aiming for correct solutions on the first attempt. Our experiments across various reasoning datasets and with different model architectures demonstrate that SMART significantly enhances the ability of models to choose optimal strategies without external guidance (+15 points on the GSM8K dataset). By achieving higher accuracy with a single inference pass, SMART not only improves performance but also reduces computational costs for refinement-based strategies, paving the way for more efficient and intelligent reasoning in LMs.
     </details>

123. **Seq-VCR: Preventing Collapse in Intermediate Transformer Representations for Enhanced Reasoning** [[pdf]](https://openreview.net/forum?id=30oIfmrcFO) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Decoder-only Transformers often struggle with complex reasoning tasks, particularly arithmetic reasoning requiring multiple sequential operations. In this work, we identify representation collapse in the model's intermediate layers as a key factor limiting their reasoning capabilities. To address this, we propose Sequential Variance-Covariance Regularization (Seq-VCR), which enhances the entropy of intermediate representations and prevents collapse. Combined with dummy pause tokens as substitutes for chain-of-thought (CoT) tokens, our method significantly improves performance in arithmetic reasoning problems. In the challenging $5 \times 5$ integer multiplication task, our approach achieves $99.5\%$ exact match accuracy, outperforming models of the same size (which yield $0\%$ accuracy) and GPT-4 with five-shot CoT prompting ($44\%$). We also demonstrate superior results on arithmetic expression and longest increasing subsequence (LIS) datasets. Our findings highlight the importance of preventing intermediate layer representation collapse to enhance the reasoning capabilities of Transformers and show that Seq-VCR offers an effective solution without requiring explicit CoT supervision.
     </details>

124. **Steering Large Language Models between Code Execution and Textual Reasoning** [[pdf]](https://arxiv.org/abs/2410.03524v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is discovered that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code, so three methods to better steer LLM code/text generation are proposed and achieve a notable improvement.
     </details>


     <details>
          <summary>Abstract</summary>
          While a lot of recent research focuses on enhancing the textual reasoning capabilities of Large Language Models (LLMs) by optimizing the multi-agent framework or reasoning chains, several benchmark tasks can be solved with 100% success through direct coding, which is more scalable and avoids the computational overhead associated with textual iterating and searching. Textual reasoning has inherent limitations in solving tasks with challenges in math, logics, optimization, and searching, which is unlikely to be solved by simply scaling up the model and data size. The recently released OpenAI GPT Code Interpreter and multi-agent frameworks such as AutoGen have demonstrated remarkable proficiency of integrating code generation and execution to solve complex tasks using LLMs. However, based on our experiments on 7 existing popular methods for steering code/text generation in both single- and multi-turn settings with 14 tasks and 6 types of LLMs (including the new O1-preview), currently there is no optimal method to correctly steer LLMs to write code when needed. We discover some interesting patterns on when models use code vs. textual reasoning with the evolution to task complexity and model sizes, which even result in an astonishingly inverse scaling law. We also discover that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code. To mitigate the above issues, we propose three methods to better steer LLM code/text generation and achieve a notable improvement. The costs of token lengths and runtime are thoroughly discussed for all the methods. We believe the problem of steering LLM code/text generation is critical for future research and has much space for further improvement. Project Page, Datasets, and Codes are available at https://yongchao98.github.io/CodeSteer/.
     </details>

125. **Step-Controlled DPO: Leveraging Stepwise Errors for Enhancing Mathematical Reasoning of Language Models** [[pdf]](https://openreview.net/forum?id=ZRDa2IT1sQ) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) has proven effective at improving the performance of large language models (LLMs) on downstream tasks such as reasoning and alignment. In this work, we propose Step-Controlled DPO (SCDPO), a method for automatically providing stepwise error supervision by creating negative samples of mathematical reasoning rationales that start making errors at a specified step. By applying these samples in DPO training, SCDPO can better align the model to avoid reasoning errors and output accurate reasoning steps. Qualitative analysis of the credit assignment of SCDPO and DPO demonstrates the effectiveness of SCDPO at identifying errors in mathematical solutions. We then apply SCDPO to an InternLM2-20B model, resulting in a 20B model that achieves competitive scores of 88.5\% on GSM8K and 58.1\% on MATH, rivaling all other open-source LLMs, showing the great potential of our method. The code, models and data are released to inspire future work.
     </details>

126. **Step-by-Step Reasoning for Math Problems via Twisted Sequential Monte Carlo** [[pdf]](http://arxiv.org/abs/2410.01920) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a novel verification method based on Twisted Sequential Monte Carlo (TSMC), which sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions.
     </details>


     <details>
          <summary>Abstract</summary>
          Augmenting the multi-step reasoning abilities of Large Language Models (LLMs) has been a persistent challenge. Recently, verification has shown promise in improving solution consistency by evaluating generated outputs. However, current verification approaches suffer from sampling inefficiencies, requiring a large number of samples to achieve satisfactory performance. Additionally, training an effective verifier often depends on extensive process supervision, which is costly to acquire. In this paper, we address these limitations by introducing a novel verification method based on Twisted Sequential Monte Carlo (TSMC). TSMC sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions. We apply TSMC to LLMs by estimating the expected future rewards at partial solutions. This approach results in a more straightforward training target that eliminates the need for step-wise human annotations. We empirically demonstrate the advantages of our method across multiple math benchmarks, and also validate our theoretical analysis of both our approach and existing verification methods.
     </details>

127. **StepProof: Step-by-step verification of natural language mathematical proofs** [[pdf]](https://openreview.net/forum?id=EXaKfdsw04) `ICLR 2025 Submission` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Interactive theorem provers (ITPs) are powerful tools for the formal verification of mathematical proofs down to the axiom level. However, their lack of a natural language interface remains a significant limitation. Recent advancements in large language models (LLMs) have enhanced the understanding of natural language inputs, paving the way for autoformalization—the process of translating natural language proofs into formal proofs that can be verified. Despite these advancements, existing autoformalization approaches are limited to verifying complete proofs and lack the capability for finer, sentence-level verification. To address this gap, we propose StepProof, a novel autoformalization method designed for granular, step-by-step verification. StepProof breaks down complete proofs into multiple verifiable subproofs, enabling sentence-level verification. Experimental results demonstrate that StepProof significantly improves proof success rates and efficiency compared to traditional methods. Additionally, we found that minor manual adjustments to the natural language proofs, tailoring them for step-level verification, further enhanced StepProof’s performance in autoformalization.
     </details>

128. **SubgoalXL: Subgoal-based Expert Learning for Theorem Proving** [[pdf]](https://arxiv.org/abs/2408.11172v1) `ICLR 2025 Submission` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SubgoalXL is introduced, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment, and achieves a new state-of-the-art performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Formal theorem proving, a field at the intersection of mathematics and computer science, has seen renewed interest with advancements in large language models (LLMs). This paper introduces SubgoalXL, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment. SubgoalXL addresses two critical challenges: the scarcity of specialized mathematics and theorem-proving data, and the need for improved multi-step reasoning abilities in LLMs. By optimizing data efficiency and employing subgoal-level supervision, SubgoalXL extracts richer information from limited human-generated proofs. The framework integrates subgoal-oriented proof strategies with an expert learning system, iteratively refining formal statement, proof, and subgoal generators. Leveraging the Isabelle environment's advantages in subgoal-based proofs, SubgoalXL achieves a new state-of-the-art performance of 56.1\% in Isabelle on the standard miniF2F dataset, marking an absolute improvement of 4.9\%. Notably, SubgoalXL successfully solves 41 AMC12, 9 AIME, and 3 IMO problems from miniF2F. These results underscore the effectiveness of maximizing limited data utility and employing targeted guidance for complex reasoning in formal theorem proving, contributing to the ongoing advancement of AI reasoning capabilities. The implementation is available at \url{https://github.com/zhaoxlpku/SubgoalXL}.
     </details>

129. **Subtle Errors Matter: Preference Learning via Error-injected Self-editing** [[pdf]](http://arxiv.org/abs/2410.06638) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel preference learning framework called RISE, which injects predefined subtle errors into partial tokens of correct solutions to construct hard pairs for error mitigation, and refines the training objective to focus on predefined errors and their tokens, without requiring fine-grained sampling or preference annotation.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited strong mathematical reasoning and computational prowess, tackling tasks ranging from basic arithmetic to advanced competition-level problems. However, frequently occurring subtle errors, such as miscalculations or incorrect substitutions, limit the models' full mathematical potential. Existing studies to improve mathematical ability typically involve distilling reasoning skills from stronger LLMs or applying preference learning to step-wise response pairs. Although these methods leverage samples of varying granularity to mitigate reasoning errors, they overlook the frequently occurring subtle errors. A major reason is that sampled preference pairs involve differences unrelated to the errors, which may distract the model from focusing on subtle errors. In this work, we propose a novel preference learning framework called eRror-Injected Self-Editing (RISE), which injects predefined subtle errors into partial tokens of correct solutions to construct hard pairs for error mitigation. In detail, RISE uses the model itself to edit a small number of tokens in the solution, injecting designed subtle errors. Then, pairs composed of self-edited solutions and their corresponding correct ones, along with pairs of correct and incorrect solutions obtained through sampling, are used together for subtle error-aware DPO training. Compared with other preference learning methods, RISE further refines the training objective to focus on predefined errors and their tokens, without requiring fine-grained sampling or preference annotation. Extensive experiments validate the effectiveness of RISE, with preference learning on Qwen2-7B-Instruct yielding notable improvements of 3.0% on GSM8K and 7.9% on MATH.
     </details>

130. **SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights** [[pdf]](http://arxiv.org/abs/2410.09008) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes SuperCorrect, a novel two-stage framework that uses a large teacher model to supervise and correct both the reasoning and reflection processes of a smaller student model and introduces cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) like GPT-4, PaLM, and LLaMA have shown significant improvements in various reasoning tasks. However, smaller models such as Llama-3-8B and DeepSeekMath-Base still struggle with complex mathematical reasoning because they fail to effectively identify and correct reasoning errors. Recent reflection-based methods aim to address these issues by enabling self-reflection and self-correction, but they still face challenges in independently detecting errors in their reasoning steps. To overcome these limitations, we propose SuperCorrect, a novel two-stage framework that uses a large teacher model to supervise and correct both the reasoning and reflection processes of a smaller student model. In the first stage, we extract hierarchical high-level and detailed thought templates from the teacher model to guide the student model in eliciting more fine-grained reasoning thoughts. In the second stage, we introduce cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training. This cross-model DPO approach teaches the student model to effectively locate and resolve erroneous thoughts with error-driven insights from the teacher model, breaking the bottleneck of its thoughts and acquiring new skills and knowledge to tackle challenging problems. Extensive experiments consistently demonstrate our superiority over previous methods. Notably, our SuperCorrect-7B model significantly surpasses powerful DeepSeekMath-7B by 7.8%/5.3% and Qwen2.5-Math-7B by 15.1%/6.3% on MATH/GSM8K benchmarks, achieving new SOTA performance among all 7B models. Code: https://github.com/YangLing0818/SuperCorrect-llm
     </details>

131. **Synthetic Theorem Generation in Lean** [[pdf]](https://openreview.net/forum?id=EeDSMy5Ruj) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The application of large language models (LLMs) to theorem proving presents a promising avenue for advancing formal mathematics. Interactive theorem provers, such as Lean, offer a rigorous framework within which these models can assist in or automate proof discovery, grounding their reasoning capabilities in a sound, verifiable formal system. However, the potential of LLMs in this domain is constrained by the limited availability of formal proof corpora for training. To address this limitation, we introduce a synthetic theorem generator capable of producing novel Lean theorems and their corresponding proofs. Our approach employs forward reasoning to synthesize new propositions from premises drawn from existing Lean libraries. We explore candidate reasoning steps using a search strategy that optimizes for diversity of output, apply them in a linear fashion that avoids irrelevant proof steps, and assess their effect by meta-programmatically executing corresponding Lean tactics. These methods enable the generation of an arbitrary number of new theorems and proofs across various mathematical domains, using common Lean proof tactics while ensuring the correctness of generated theorems by construction.  We demonstrate the efficacy of the generated theorems and training data by fine-tuning models on synthetic theorems and evaluating them on the miniF2F-test benchmark. Our results show improvements in theorem-proving capabilities, with accuracy increasing from 37.3% to 38.5% for the Falcon2-11B model trained solely on Mathlib, and from 38.1% to 39.3% for the same model trained on a mix of rich datasets. These improvements highlight the value of our diverse synthetic data in augmenting limited existing corpora of formal proofs, providing complementary information that enhances LLMs' performance on theorem-proving tasks even when combined with other datasets.
     </details>

132. **TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees** [[pdf]](http://arxiv.org/abs/2410.12854) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In the domain of complex reasoning tasks, such as mathematical reasoning, recent advancements have proposed the use of Direct Preference Optimization (DPO) to suppress output of dispreferred responses, thereby enhancing the long-chain reasoning capabilities of large language models (LLMs). To this end, these studies employed LLMs to generate preference trees via Tree-of-thoughts (ToT) and sample the paired preference responses required by the DPO algorithm. However, the DPO algorithm based on binary preference optimization is unable to learn multiple responses with varying degrees of preference/dispreference that provided by the preference trees, resulting in incomplete preference learning. In this work, we introduce Tree Preference Optimization (TPO), that does not sample paired preference responses from the preference tree; instead, it directly learns from the entire preference tree during the fine-tuning. Specifically, TPO formulates the language model alignment as a Preference List Ranking problem, where the policy can potentially learn more effectively from a ranked preference list of responses given the prompt. In addition, to further assist LLMs in identifying discriminative steps within long-chain reasoning and increase the relative reward margin in the preference list, TPO utilizes Adaptive Step Reward to adjust the reward values of each step in trajectory for performing fine-grained preference optimization. We carry out extensive experiments on mathematical reasoning tasks to evaluate TPO. The experimental results indicate that TPO consistently outperforms DPO across three public large language models on four datasets.
     </details>

133. **The Perfect Blend: Redefining RLHF with Mixture of Judges** [[pdf]](http://arxiv.org/abs/2409.20370) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel post-training paradigm which can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives is introduced, called Constrained Generative Policy Optimization (CGPO).
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement learning from human feedback (RLHF) has become the leading approach for fine-tuning large language models (LLM). However, RLHF has limitations in multi-task learning (MTL) due to challenges of reward hacking and extreme multi-objective optimization (i.e., trade-off of multiple and/or sometimes conflicting objectives). Applying RLHF for MTL currently requires careful tuning of the weights for reward model and data combinations. This is often done via human intuition and does not generalize. In this work, we introduce a novel post-training paradigm which we called Constrained Generative Policy Optimization (CGPO). The core of CGPO is Mixture of Judges (MoJ) with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. It shows strong empirical results with theoretical guarantees, does not require extensive hyper-parameter tuning, and is plug-and-play in common post-training pipelines. Together, this can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives.   Our empirical evaluations demonstrate that CGPO significantly outperforms standard RLHF algorithms like PPO and DPO across various tasks including general chat, STEM questions, instruction following, and coding. Specifically, CGPO shows improvements of 7.4% in AlpacaEval-2 (general chat), 12.5% in Arena-Hard (STEM & reasoning), and consistent gains in other domains like math and coding. Notably, PPO, while commonly used, is prone to severe reward hacking in popular coding benchmarks, which CGPO successfully addresses. This breakthrough in RLHF not only tackles reward hacking and extreme multi-objective optimization challenges but also advances the state-of-the-art in aligning general-purpose LLMs for diverse applications.
     </details>

134. **The Role of Deductive and Inductive Reasoning in Large Language Models** [[pdf]](https://arxiv.org/abs/2410.02892v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Deductive and InDuctive method is proposed, which enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning within the prompt construction process, and provides a more robust and cognitively aligned framework for reasoning in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have achieved substantial progress in artificial intelligence, particularly in reasoning tasks. However, their reliance on static prompt structures, coupled with limited dynamic reasoning capabilities, often constrains their adaptability to complex and evolving problem spaces. In this paper, we propose the Deductive and InDuctive(DID) method, which enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning within the prompt construction process. Drawing inspiration from cognitive science, the DID approach mirrors human adaptive reasoning mechanisms, offering a flexible framework that allows the model to adjust its reasoning pathways based on task context and performance. We empirically validate the efficacy of DID on established datasets such as AIW and MR-GSM8K, as well as on our custom dataset, Holiday Puzzle, which presents tasks about different holiday date calculating challenges. By leveraging DID's hybrid prompt strategy, we demonstrate significant improvements in both solution accuracy and reasoning quality, achieved without imposing substantial computational overhead. Our findings suggest that DID provides a more robust and cognitively aligned framework for reasoning in LLMs, contributing to the development of advanced LLM-driven problem-solving strategies informed by cognitive science models.
     </details>

135. **Token-Supervised Value Models for Enhancing Mathematical Problem-Solving Capabilities of Large Language Models** [[pdf]](https://openreview.net/forum?id=6HcnC3pPkp) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          With the rapid advancement of test-time compute search strategies to improve the mathematical problem-solving capabilities of large language models (LLMs), the need for building robust verifiers has become increasingly important. However, all these inference strategies rely on existing verifiers originally designed for Best-of-N search, which makes them sub-optimal for tree search techniques at test time. During tree search, existing verifiers can only offer indirect and implicit assessments of partial solutions or under-value prospective intermediate steps, thus resulting in the premature pruning of promising intermediate steps. To overcome these limitations, we propose token-supervised value models (TVMs) -- a new class of verifiers that assign each token a probability that reflects the likelihood of reaching the correct final answer. This new token-level supervision enables TVMs to directly and explicitly evaluate partial solutions, effectively distinguishing between promising and incorrect intermediate steps during tree search at test time. Experimental results demonstrate that combining tree-search-based inference strategies with TVMs significantly improves the accuracy of LLMs in mathematical problem-solving tasks, surpassing the performance of existing verifiers.
     </details>

136. **Towards Geometry Problems Solving Employing GPT-4 Vision with Few-Shot Prompting: An Empirical Study of What Matters** [[pdf]](https://openreview.net/forum?id=0vKokoPKTo) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The few demonstrations ("few-shot prompting") can significantly improve the ability of Large Language Models (LLMs) in mathematical reasoning, including geometry problem solving (GPS).  GPT-4 Vision (GPT-4V), as a leading example of LLMs, also demonstrates significant improvements.  This tremendous achievement is mainly attributed to prompting methods like "Chain-of-Thought" and "Program-of-Thought," which leverage the in-context learning ability of the model combined with few-shot prompting to solve new problems.  Despite the success of these prompting methods, it remains understood what the GPT-4V model learns from the demonstrations that lead to improved performance.  In this paper, we evaluated the answering accuracy of GPT-4V with 2-shot prompting on five geometric problem datasets and conducted a series of detailed analyses.  Firstly, through ablation experiments with valid and invalid demonstration examples, we found that the model’s performance improvement is not due to the quality of the demonstration, but rather to the input format, output format, and logic and structure of the demonstration.  Secondly, by analyzing the reasoning and computational requirements of geometric problems, and verifying experimental results, we found that GPS tasks emphasize reasoning ability more than computational power.  Finally, our analysis of various prompt methods revealed that existing approaches are not effective at improving model performance concerning problem length and geometric shape.  Therefore, specialized prompt methods could be designed to enhance the model's performance in these aspects, or fine-tuning the model by adding problem data with longer lengths or mixed geometric shapes could optimize its performance.  Overall, developing an LLM that fully adapts to GPS tasks represents a key research direction.  The source code and data will be made available in a GitHub repository.
     </details>

137. **TypedThinker: Typed Thinking Improves Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2410.01952) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          TypedThinker is a novel framework that enhances LLMs' problem-solving abilities by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical) and shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in the reasoning capabilities of Large Language Models (LLMs), the lack of diverse reasoning solutions often makes them trapped in a limited solution search area. In this paper, we propose TypedThinker, a novel framework that enhances LLMs' problem-solving abilities by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical). Our analysis across four benchmarks reveals that different reasoning types uniquely solve distinct sets of problems, highlighting the importance of diverse thinking approaches. TypedThinker addresses two key challenges: selecting appropriate reasoning types for given problems and effectively implementing specific reasoning types. Through self-training on successful experiences, TypedThinker learns an implicit policy for reasoning type selection and application. Experimental results demonstrate significant improvements over baseline models, with accuracy increases of 3.4% for Mistral 7B and 16.7% for LLaMA3 8B across four reasoning benchmarks. Notably, TypedThinker shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o. The code is released at https://github.com/dqwang122/ThinkHub.
     </details>

138. **U-MATH: A University-Level Benchmark for Evaluating Mathematical Skills in LLMs** [[pdf]](https://openreview.net/forum?id=xlxGsX1pc7) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The current evaluation of mathematical skills in LLMs is limited, as existing benchmarks are relatively small, primarily focus on elementary and high-school problems, or lack diversity in topics. Additionally, the inclusion of visual elements in tasks remains largely under-explored.       To address these gaps, we introduce **U-MATH**, a novel benchmark of 1,125 unpublished open-ended university-level problems sourced from teaching materials. It is balanced across six core subjects, with 20\% of problems requiring image understanding. Given the open-ended nature of U-MATH problems, we employ an LLM to judge the correctness of generated solutions. To this end, we release **$\boldsymbol\mu$-MATH**, an additional dataset to evaluate the LLMs' capabilities in assessing solutions.  The evaluation of general domain, math-specific, and multimodal LLMs highlights the challenges presented by U-MATH. Our findings reveal that LLMs achieve a maximum accuracy of only 53\% on text-based tasks, with even lower 30\% on visual problems. The solution assessment proves challenging for LLMs, with the best LLM judge having an F1-score of 76\% on $\mu$-MATH.      We open-source U-MATH, $\mu$-MATH, and evaluation code on GitHub.
     </details>

139. **UGMathBench: A Diverse and Dynamic Benchmark for Undergraduate-Level Mathematical Reasoning with Large Language Models** [[pdf]](https://openreview.net/forum?id=fovPyqPcKY) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have made significant strides in mathematical reasoning, underscoring the need for a comprehensive and fair evaluation of their capabilities. However, existing benchmarks often fall short, either lacking extensive coverage of undergraduate-level mathematical problems or probably suffering from test-set contamination. To address these issues, we introduce UGMathBench, a diverse and dynamic benchmark specifically designed for evaluating undergraduate-level mathematical reasoning with LLMs. UGMathBench comprises 5,062 problems across 16 subjects and 111 topics, featuring 10 distinct answer types. Each problem includes three randomized versions, with additional versions planned for release as the leading open-source LLMs become saturated in UGMathBench. Furthermore, we propose two key metrics: effective accuracy (EAcc), which measures the percentage of correctly solved problems across all three versions, and reasoning gap ($\Delta$), which assesses reasoning robustness by calculating the difference between the average accuracy across all versions and EAcc. Our extensive evaluation of 23 leading LLMs reveals that the highest EAcc achieved is 56.3% by OpenAI-o1-mini, with large $\Delta$ values observed across different models. This highlights the need for future research aimed at developing "large reasoning models" with high EAcc and $\Delta = 0$. We anticipate that the release of UGMathBench, along with its detailed evaluation codes, will serve as a valuable resource to advance the development of LLMs in solving mathematical problems.
     </details>

140. **Understanding Reasoning with Looped Models** [[pdf]](https://openreview.net/forum?id=din0lGfZFd) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models have shown promising abilities in reasoning problems and scaling laws suggest that parameter count is a key driver. Recent works (Chen & Zou, 2024; Ye et al., 2024) argue that for reasoning, depth plays a very important role in addition to parameter count. In this work, we make a more fine-grained claim — many reasoning problems require large depth but not necessarily many parameters, in the sense that they can be solved via looped models. This unlocks a novel application of looped models for reasoning. We empirically study various synthetic reasoning problems like addition, variable assignment and math problems. For each of these, we find that $k$-layer transformer model looped $L$ times nearly matches the quality of a $kL$-layer non-looped model and is much better than a k-layer model. Thus, using a small model and providing depth via looping can suffice for such reasoning problems. We then show theoretical results proving that many such reasoning problems can be solved via iterative algorithms, and thus, can be solved with looped models. Motivated by these findings, we train autoregressive models on general language modeling datasets with looping and compare a $k$-layer model looped $L$ times to a $kL$-layer model. While the looped model is understandably worse on perplexity and memorization tasks, it surprisingly does very well on tasks that require reasoning, like open book QA, math word problems and reasoning primitives. Despite having significantly fewer parameters, it can even match or outperform the non-looped $kL$-layer model on some of these tasks. These results suggest a novel inductive bias of looped models towards enhanced reasoning. We provide further evidence for this inductive bias by visualizing perplexity vs downstream isoplots, and design a looping-inspired regularization that solidifies this hypothesis.
     </details>

141. **Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch** [[pdf]](http://arxiv.org/abs/2410.18693) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The availability of high-quality data is one of the most important factors in improving the reasoning capability of LLMs. Existing works have demonstrated the effectiveness of creating more instruction data from seed questions or knowledge bases. Recent research indicates that continually scaling up data synthesis from strong models (e.g., GPT-4) can further elicit reasoning performance. Though promising, the open-sourced community still lacks high-quality data at scale and scalable data synthesis methods with affordable costs. To address this, we introduce ScaleQuest, a scalable and novel data synthesis method that utilizes "small-size" (e.g., 7B) open-source models to generate questions from scratch without the need for seed data with complex augmentation constraints. With the efficient ScaleQuest, we automatically constructed a mathematical reasoning dataset consisting of 1 million problem-solution pairs, which are more effective than existing open-sourced datasets. It can universally increase the performance of mainstream open-source models (i.e., Mistral, Llama3, DeepSeekMath, and Qwen2-Math) by achieving 29.2% to 46.4% gains on MATH. Notably, simply fine-tuning the Qwen2-Math-7B-Base model with our dataset can even surpass Qwen2-Math-7B-Instruct, a strong and well-aligned model on closed-source data, and proprietary models such as GPT-4-Turbo and Claude-3.5 Sonnet.
     </details>

142. **Unlocking Reasoning Potential in Large Language Models by Scaling Code-form Planning** [[pdf]](https://openreview.net/forum?id=dCPF1wlqj8) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite the remarkable success of large language models (LLMs) on traditional natural language processing tasks, their planning ability remains a critical bottleneck in tackling complex multi-step reasoning tasks. Existing approaches mainly rely on prompting or task-specific fine-tuning, often suffering from weak robustness and cross-task generalization. To address the limitation, we introduce CodePlan, a scalable paradigm that empowers LLMs to generate and follow code-form plans---pseudocode that outlines high-level, structured reasoning processes. By leveraging the structured and versatile nature of code, CodePlan effectively captures the rich semantics and control flows inherent to sophisticated reasoning. Importantly, CodePlan allows the automatic extraction of code-form plans from massive, wide-ranging text corpora without the need for curated, task-specific datasets. This enables it to scale up efficiently and improve reasoning capabilities across diverse scenarios. To train CodePlan, we construct a large-scale dataset of 2M examples that integrate code-form plans with standard prompt-response pairs from existing corpora. With minimal computation overhead during both training and inference, CodePlan achieves a 25.1\% relative improvement compared with directly generating responses, averaged across 13 challenging multi-step reasoning benchmarks, spanning mathematical reasoning, symbolic reasoning, instruction-following, multi-hop QA, and decision-making tasks. Further analysis reveals CodePlan's increasing performance gains on more complex reasoning tasks, as well as significant data efficiency thanks to its generalization ability.
     </details>

143. **Unlocking Structured Thinking in Language Models with Cognitive Prompting** [[pdf]](https://arxiv.org/abs/2410.02953v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Cognitive prompting, particularly when dynamically adapted, significantly improves the performance of larger models, such as LLaMA3.1 70B, and enhances their ability to handle multi-step reasoning tasks, highlighting cognitive prompting as a promising strategy for general-purpose AI reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose cognitive prompting as a novel approach to guide problem-solving in large language models (LLMs) through structured, human-like cognitive operations such as goal clarification, decomposition, filtering, abstraction, and pattern recognition. By employing systematic, step-by-step reasoning, cognitive prompting enables LLMs to efficiently tackle complex, multi-step tasks. We evaluate the effectiveness of cognitive prompting on Meta's LLaMA models, comparing performance on arithmetic reasoning tasks using the GSM8K dataset and on commonsense reasoning benchmarks. Our analysis includes comparisons between models without cognitive prompting, models with a static sequence of cognitive operations, and models using reflective cognitive prompting, where the LLM dynamically self-selects the sequence of cognitive operations. The results show that cognitive prompting, particularly when dynamically adapted, significantly improves the performance of larger models, such as LLaMA3.1 70B, and enhances their ability to handle multi-step reasoning tasks. This approach also improves interpretability and flexibility, highlighting cognitive prompting as a promising strategy for general-purpose AI reasoning.
     </details>

144. **VerifierQ: Enhancing LLM Test Time Compute with Q-Learning-based Verifiers** [[pdf]](https://openreview.net/forum?id=OD9pwKQzXl) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          VerifierQ is among the first to investigate the verifier (critic) aspect in LLMs through Q-learning, and introduces a modified Bellman update for bounded Q-values, incorporates Implicit Q-learning (IQL) for efficient action space management, and integrates a novel Conservative Q-learning (CQL) formulation for balanced Q-value estimation.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in test time compute, particularly through the use of verifier models, have significantly enhanced the reasoning capabilities of Large Language Models (LLMs). This generator-verifier approach closely resembles the actor-critic framework in reinforcement learning (RL). However, current verifier models in LLMs often rely on supervised fine-tuning without temporal difference learning such as Q-learning. This paper introduces VerifierQ, a novel approach that integrates Offline Q-learning into LLM verifier models. We address three key challenges in applying Q-learning to LLMs: (1) handling utterance-level Markov Decision Processes (MDPs), (2) managing large action spaces, and (3) mitigating overestimation bias. VerifierQ introduces a modified Bellman update for bounded Q-values, incorporates Implicit Q-learning (IQL) for efficient action space management, and integrates a novel Conservative Q-learning (CQL) formulation for balanced Q-value estimation. Our method enables parallel Q-value computation and improving training efficiency. While recent work has explored RL techniques like MCTS for generators, VerifierQ is among the first to investigate the verifier (critic) aspect in LLMs through Q-learning. This integration of RL principles into verifier models complements existing advancements in generator techniques, potentially enabling more robust and adaptive reasoning in LLMs. Experimental results on mathematical reasoning tasks demonstrate VerifierQ's superior performance compared to traditional supervised fine-tuning approaches, with improvements in efficiency, accuracy and robustness. By enhancing the synergy between generation and evaluation capabilities, VerifierQ contributes to the ongoing evolution of AI systems in addressing complex cognitive tasks across various domains.
     </details>

145. **VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment** [[pdf]](http://arxiv.org/abs/2410.01679) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          VinePPO is proposed, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates, bypassing the need for large value networks, and consistently outperforms PPO and other RL-free baselines across MATH and GSM8K datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is essential for enhancing model performance. Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning (RL) algorithm used for LLM finetuning, employs value networks to tackle credit assignment. However, value networks face challenges in predicting the expected cumulative rewards accurately in complex reasoning tasks, often leading to high-variance updates and suboptimal performance. In this work, we systematically evaluate the efficacy of value networks and reveal their significant shortcomings in reasoning-heavy LLM tasks, showing that they barely outperform a random baseline when comparing alternative steps. To address this, we propose VinePPO, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates, bypassing the need for large value networks. Our method consistently outperforms PPO and other RL-free baselines across MATH and GSM8K datasets with fewer gradient updates (up to 9x), less wall-clock time (up to 3.0x). These results emphasize the importance of accurate credit assignment in RL finetuning of LLM and demonstrate VinePPO's potential as a superior alternative.
     </details>

146. **VisScience: An Extensive Benchmark for Evaluating K12 Educational Multi-modal Scientific Reasoning** [[pdf]](http://arxiv.org/abs/2409.13730) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive benchmark is constructed, named VisScience, which is utilized to assess the multi-modal scientific reasoning across the three disciplines of mathematics, physics, and chemistry and demonstrates that closed-source MLLMs generally outperform open-source models.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal large language models (MLLMs) have demonstrated promising capabilities across various tasks by integrating textual and visual information to achieve visual understanding in complex scenarios. Despite the availability of several benchmarks aims to evaluating MLLMs in tasks from visual question answering to complex problem-solving, most focus predominantly on mathematics or general visual understanding tasks. This reveals a critical gap in current benchmarks, which often overlook the inclusion of other key scientific disciplines such as physics and chemistry. To address this gap, we meticulously construct a comprehensive benchmark, named VisScience, which is utilized to assess the multi-modal scientific reasoning across the three disciplines of mathematics, physics, and chemistry. This benchmark comprises 3,000 questions drawn from K12 education - spanning elementary school through high school - equally distributed across three disciplines, with 1,000 questions per discipline. The questions within VisScience span 21 distinct subjects and are categorized into five difficulty levels, offering a broad spectrum of topics within each discipline. With VisScience, we present a detailed evaluation of the performance of 25 representative MLLMs in scientific reasoning. Experimental results demonstrate that closed-source MLLMs generally outperform open-source models. The best performance observed include a 53.4\% accuracy in mathematics by Claude3.5-Sonnet, 38.2\% in physics by GPT-4o, and 47.0\% in chemistry by Gemini-1.5-Pro. These results underscore the strengths and limitations of MLLMs, suggesting areas for future improvement and highlighting the importance of developing models that can effectively handle the diverse demands of multi-modal scientific reasoning.
     </details>

147. **WISDOM: Progressive Curriculum Synthesis Makes LLMs Better Mathematical Reasoner** [[pdf]](https://openreview.net/forum?id=hFFAg5Dmw9) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of problem-solving tasks. Despite their success, LLMs still face significant challenges in complex reasoning, particularly with advanced mathematical problems. These problems require not only a deep understanding of task descriptions but also sophisticated logical and mathematical reasoning to determine the correct solution path, which is often lacking in the existing synthetic data. To address this gap, we introduce WISDOM, which draws inspiration from the human learning process and employs curriculum learning to gradually synthesize high-quality CoT data from easy to hard. Our goal is to guide LLM training and improve reasoning capabilities by progressively exposing models to increasingly challenging problems. Based on the synthesized data, we further fine-tune and develop the WISDOM series models, achieving significant improvements across multiple mathematical reasoning benchmarks. Notably, WISDOM-7B (DSMath) achieves a score of 62.4% on MATH, matching GPT-4’s performance with 2/30 correct answers on AIME2024. Furthermore, WISDOM-70B (Llama3) outperforms GPT-4 on AIME2024 with 3/30 correct answers, demonstrating its potential as a better mathematical reasoner. More data and models will be available at https://anonymous.4open.science/r/Wisdom-math-377B
     </details>

148. **What Kind of Pretraining Data Do Large Language Models Rely on When Doing Reasoning?** [[pdf]](https://openreview.net/forum?id=1hQKHHUsMx) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The capabilities and limitations of Large Language Models (LLMs) have been sketched out in great detail in recent years, providing an intriguing yet conflicting picture. On the one hand, LLMs demonstrate a general ability to solve problems. On the other hand, they show surprising reasoning gaps when compared to humans, casting doubt on the robustness of their generalisation strategies. The sheer volume of data used in the design of LLMs has precluded us from applying the method traditionally used to measure generalisation; train-test set separation. In this work, we study what kind of generalisation strategies LLMs employ when performing reasoning tasks by investigating the pretraining data they rely on. For two models of different sizes (7B and 35B) and 2.5B of their pretraining tokens, we identify what documents impact three simple mathematical reasoning tasks and contrast this to the data that are influential for answering factual questions. We find that, while the models rely on mostly distinct sets of data for each factual question, documents often have a similar influence on different reasoning questions with the same task, indicating the presence of procedural knowledge. We further find that the answers to the factual questions often show up in the most influential data. However, for the reasoning questions the answers usually do not show up as highly influential, nor do the answers to the intermediate reasoning steps. When we characterise the top portion of the ranking for the reasoning questions qualitatively, we find that the influential documents often contain procedural knowledge, like demonstrating how to obtain the solution using formulae or code. Our findings indicate that the generalisation strategy the model uses when doing reasoning is unlike retrieval, but more like a strategy using many documents doing a similar form of reasoning.
     </details>

149. **What Makes Large Language Models Reason in (Multi-Turn) Code Generation?** [[pdf]](http://arxiv.org/abs/2410.08105) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study investigates the effects of a wide range of prompting strategies with a focus on automatic re-prompting over multiple turns and computational requirements, and reveals strategies that consistently improve performance across all models with small and large sampling budgets.
     </details>


     <details>
          <summary>Abstract</summary>
          Prompting techniques such as chain-of-thought have established themselves as a popular vehicle for improving the outputs of large language models (LLMs). For code generation, however, their exact mechanics and efficacy are under-explored. We thus investigate the effects of a wide range of prompting strategies with a focus on automatic re-prompting over multiple turns and computational requirements. After systematically decomposing reasoning, instruction, and execution feedback prompts, we conduct an extensive grid search on the competitive programming benchmarks CodeContests and TACO for multiple LLM families and sizes (Llama 3.0 and 3.1, 8B, 70B, 405B, and GPT-4o). Our study reveals strategies that consistently improve performance across all models with small and large sampling budgets. We then show how finetuning with such an optimal configuration allows models to internalize the induced reasoning process and obtain improvements in performance and scalability for multi-turn code generation.
     </details>

150. **ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment** [[pdf]](https://openreview.net/forum?id=4JBEpP6eRS) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Data selection is crucial for optimizing language model (LM) performance on specific tasks, yet most existing methods fail to effectively consider the target task distribution.   Current approaches either ignore task-specific requirements entirely or rely on approximations that fail to capture the nuanced patterns needed for tasks like Autoformalization or code generation.   Methods that do consider the target distribution often rely on simplistic, sometimes noisy, representations, like hashed n-gram features, which can lead to collisions and introduce noise.   We introduce ZIP-FIT, a data selection framework that uses gzip compression to directly measure alignment between potential training data and the target task distribution.   In extensive evaluations on Autoformalization and Python code generation, ZIP-FIT significantly outperforms leading baselines like DSIR and D4.   Models trained on ZIP-FIT-selected data achieve their lowest cross-entropy loss up to 85.1\% faster than baselines, demonstrating that better task alignment leads to more efficient learning.   In addition, ZIP-FIT performs selection up to 65.8\% faster than DSIR and two orders of magnitude faster than D4.   Notably, ZIP-FIT shows that smaller, well-aligned datasets often outperform larger but less targeted ones, demonstrating that a small amount of higher quality data is superior to a large amount of lower quality data.   Our results imply that task-aware data selection is crucial for efficient domain adaptation, and that compression offers a principled way to measure task alignment.   By showing that targeted data selection can dramatically improve task-specific performance, our work provides new insights into the relationship between data quality, task alignment, and model learning efficiency.
     </details>

