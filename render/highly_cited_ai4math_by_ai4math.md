# Highly Cited AI4Math by AI4Math 



1. **Training Verifiers to Solve Math Word Problems** [[pdf]](http://arxiv.org/abs/2110.14168) `2021-11-17` (1000 cite) (128 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that verification significantly improves performance on GSM8K, and there is strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline.
     </details>


     <details>
          <summary>Abstract</summary>
          State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. To increase performance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that verification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline.
     </details>

2. **Self-Consistency Improves Chain of Thought Reasoning in Language Models** [[pdf]](http://arxiv.org/abs/2203.11171) `ICLR 2023` (1000 cite) (88 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting that first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting combined with pretrained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out all possible reasoning paths. Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer. Our extensive empirical evaluation shows that self-consistency boosts the performance of chain-of-thought prompting with a striking margin on a range of popular arithmetic and commonsense reasoning benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).
     </details>

3. **Measuring Mathematical Problem Solving With the MATH Dataset** [[pdf]](http://arxiv.org/abs/2103.03874) `NeurIPS 2021` (935 cite) (82 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces MATH, a new dataset of 12,500 challenging competition mathematics problems which can be used to teach models to generate answer derivations and explanations and shows that accuracy remains relatively low, even with enormous Transformer models.
     </details>


     <details>
          <summary>Abstract</summary>
          Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. To facilitate future research and increase accuracy on MATH, we also contribute a large auxiliary pretraining dataset which helps teach models the fundamentals of mathematics. Even though we are able to increase accuracy on MATH, our results show that accuracy remains relatively low, even with enormous Transformer models. Moreover, we find that simply increasing budgets and model parameter counts will be impractical for achieving strong mathematical reasoning if scaling trends continue. While scaling Transformers is automatically solving most other text-based tasks, scaling is not currently solving MATH. To have more traction on mathematical problem solving we will likely need new algorithmic advancements from the broader research community.
     </details>

4. **Large Language Models are Zero-Shot Reasoners** [[pdf]](https://arxiv.org/abs/2205.11916v4) `NeurIPS 2022 Poster` (1000 cite) (74 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results demonstrate that the Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics, symbolic reasoning, and other logical reasoning tasks, without any hand-crafted few-shot examples.
     </details>


     <details>
          <summary>Abstract</summary>
          Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-by-step answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs' ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding ``Let's think step by step'' before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large-scale InstructGPT model (text-davinci-002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars.
     </details>

5. **Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems** [[pdf]](https://aclanthology.org/P17-1015) `ACL 2017 Long Papers` (543 cite) (65 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving algebraic word problems requires executing a series of arithmetic operations—a program—to obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.
     </details>

6. **Are NLP Models really able to Solve Simple Math Word Problems?** [[pdf]](http://arxiv.org/abs/2103.07191) `NAACL 2021 Main` (574 cite) (63 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs, and models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          The problem of designing NLP solvers for math word problems (MWP) has seen sustained research activity and steady gains in the test accuracy. Since existing solvers achieve high performance on the benchmark datasets for elementary level MWPs containing one-unknown arithmetic word problems, such problems are often considered “solved” with the bulk of research attention moving to more complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower. We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high performance on the benchmark datasets. To this end, we show that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs. Similarly, models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we introduce a challenge dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing datasets. The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much remains to be done even for the simplest of the MWPs.
     </details>

7. **Solving General Arithmetic Word Problems** [[pdf]](https://aclanthology.org/D15-1202) `2015-09-01` (385 cite) (56 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This is the first algorithmic approach that can handle arithmetic problems with multiple steps and operations, without depending on additional annotations or predefined templates, and it outperforms existing systems, achieving state of the art performance on benchmark datasets of arithmetic word problems.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

8. **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2211.12588) `2023-10-22` (524 cite) (53 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Under both few-shot and zero-shot settings, PoT can show an average performance gain over CoT by around 12\% across all the evaluated datasets, and by combining PoT with self-consistency decoding, can achieve SoTA performance on all math problem datasets and near-SoTAperformance on financial datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, there has been significant progress in teaching language models to perform step-by-step reasoning to solve complex numerical reasoning tasks. Chain-of-thoughts prompting (CoT) is by far the state-of-art method for these tasks. CoT uses language models to perform both reasoning and computation in the multi-step `thought' process. To disentangle computation from reasoning, we propose `Program of Thoughts' (PoT), which uses language models (mainly Codex) to express the reasoning process as a program. The computation is relegated to an external computer, which executes the generated programs to derive the answer. We evaluate PoT on five math word problem datasets (GSM, AQuA, SVAMP, TabMWP, MultiArith) and three financial-QA datasets (FinQA, ConvFinQA, TATQA) for both few-shot and zero-shot setups. Under both few-shot and zero-shot settings, PoT can show an average performance gain over CoT by around 12\% across all the evaluated datasets. By combining PoT with self-consistency decoding, we can achieve SoTA performance on all math problem datasets and near-SoTA performance on financial datasets. All of our data and code are released in Github https://github.com/wenhuchen/Program-of-Thoughts
     </details>

9. **MAWPS: A Math Word Problem Repository** [[pdf]](https://aclanthology.org/N16-1136) `NAACL 2016 Main` (308 cite) (53 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          MAWPS allows for the automatic construction of datasets with particular characteristics, providing tools for tuning the lexical and template overlap of a dataset as well as for filtering ungrammatical problems from web-sourced corpora.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

10. **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** [[pdf]](https://openreview.net/forum?id=WZH7099tgfM) `ICLR 2023 Poster` (788 cite) (49 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difficult problems than those seen in the prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting has demonstrated remarkable performance on various natural language reasoning tasks. However, it tends to perform poorly on tasks which requires solving problems harder than the exemplars shown in the prompts. To overcome this challenge of easy-to-hard generalization, we propose a novel prompting strategy, least-to-most prompting. The key idea in this strategy is to break down a complex problem into a series of simpler subproblems and then solve them in sequence. Solving each subproblem is facilitated by the answers to previously solved subproblems. Our experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difficult problems than those seen in the prompts. A notable finding is that when the GPT-3 code-davinci-002 model is used with least-to-most prompting, it can solve the compositional generalization benchmark SCAN in any split (including length split) with an accuracy of at least 99\% using just 14 exemplars, compared to only 16\% accuracy with chain-of-thought prompting. This is particularly noteworthy because neural-symbolic models in the literature that specialize in solving SCAN are trained on the entire training set containing over 15,000 examples. We have included prompts for all the tasks in the Appendix.
     </details>

11. **Solving Quantitative Reasoning Problems with Language Models** [[pdf]](http://arxiv.org/abs/2206.14858) `NeurIPS 2022 Poster` (553 cite) (46 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language models have achieved remarkable performance on a wide range of tasks that require natural language understanding. Nevertheless, state-of-the-art models have generally struggled with tasks that require quantitative reasoning, such as solving mathematics, science, and engineering questions at the college level. To help close this gap, we introduce Minerva, a large language model pretrained on general natural language data and further trained on technical content. The model achieves strong performance in a variety of evaluations, including state-of-the-art performance on the MATH dataset. We also evaluate our model on over two hundred undergraduate-level problems in physics, biology, chemistry, economics, and other sciences that require quantitative reasoning, and find that the model can correctly answer nearly a quarter of them.
     </details>

12. **Learning to Solve Arithmetic Word Problems with Verb Categorization** [[pdf]](https://aclanthology.org/D14-1058) `2014-10-01` (343 cite) (43 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The paper analyzes the arithmetic-word problems “genre”, identifying seven categories of verbs used in such problems, and reports the first learning results on this task without reliance on predefined templates and makes the data publicly available.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

13. **MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms** [[pdf]](http://arxiv.org/abs/1905.13319) `NAACL 2019 Main` (393 cite) (42 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A large-scale dataset of math word problems and an interpretable neural math problem solver by learning to map problems to their operation programs and a new representation language to model operation programs corresponding to each math problem that aim to improve both the performance and the interpretability of the learned models.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce a large-scale dataset of math word problems and an interpretable neural math problem solver by learning to map problems to their operation programs. Due to annotation challenges, current datasets in this domain have been either relatively small in scale or did not offer precise operational annotations over diverse problem types. We introduce a new representation language to model operation programs corresponding to each math problem that aim to improve both the performance and the interpretability of the learned models. Using this representation language, we significantly enhance the AQUA-RAT dataset with fully-specified operational programs. We additionally introduce a neural sequence-to-program model with automatic problem categorization. Our experiments show improvements over competitive baselines in our dataset as well as the AQUA-RAT dataset. The results are still lower than human performance indicating that the dataset poses new challenges for future research. Our dataset is available at https://math-qa.github.io/math-QA/
     </details>

14. **Deep Neural Solver for Math Word Problems** [[pdf]](https://aclanthology.org/D17-1088) `EMNLP 2017 Main` (310 cite) (42 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experiments conducted on a large dataset show that the RNN model and the hybrid model significantly outperform state-of-the-art statistical learning methods for math word problem solving.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a deep neural solver to automatically solve math word problems. In contrast to previous statistical learning approaches, we directly translate math word problems to equation templates using a recurrent neural network (RNN) model, without sophisticated feature engineering. We further design a hybrid model that combines the RNN model and a similarity-based retrieval model to achieve additional performance improvement. Experiments conducted on a large dataset show that the RNN model and the hybrid model significantly outperform state-of-the-art statistical learning methods for math word problem solving.
     </details>

15. **Generative Language Modeling for Automated Theorem Proving** [[pdf]](https://arxiv.org/abs/2009.03393) `N/A` (234 cite) (39 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents an automated prover and proof assistant, GPT-f, for the Metamath formalization language, and analyzes its performance, finding new short proofs that were accepted into the mainMetamath library, which is to this knowledge, the first time a deep-learning based system has contributed proofs that are adopted by a formal mathematics community.
     </details>


     <details>
          <summary>Abstract</summary>
          We explore the application of transformer-based language models to automated theorem proving. This work is motivated by the possibility that a major limitation of automated theorem provers compared to humans -- the generation of original mathematical terms -- might be addressable via generation from language models. We present an automated prover and proof assistant, GPT-f, for the Metamath formalization language, and analyze its performance. GPT-f found new short proofs that were accepted into the main Metamath library, which is to our knowledge, the first time a deep-learning based system has contributed proofs that were adopted by a formal mathematics community.
     </details>

16. **Learning to Automatically Solve Algebra Word Problems** [[pdf]](https://aclanthology.org/P14-1026) `ACL 2014 Long Papers` (347 cite) (39 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An approach for automatically learning to solve algebra word problems by reasons across sentence boundaries to construct and solve a system of linear equations, while simultaneously recovering an alignment of the variables and numbers to the problem text.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

17. **A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers** [[pdf]](http://arxiv.org/abs/2106.15772) `ACL 2020` (253 cite) (37 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A metric to measure the lexicon usage diversity of a given MWP corpus is proposed, and it is demonstrated that ASDiv (Academia Sinica Diverse MWP Dataset) is more diverse than existing corpora.
     </details>


     <details>
          <summary>Abstract</summary>
          We present ASDiv (Academia Sinica Diverse MWP Dataset), a diverse (in terms of both language patterns and problem types) English math word problem (MWP) corpus for evaluating the capability of various MWP solvers. Existing MWP corpora for studying AI progress remain limited either in language usage patterns or in problem types. We thus present a new English MWP corpus with 2,305 MWPs that cover more text patterns and most problem types taught in elementary school. Each MWP is annotated with its problem type and grade level (for indicating the level of difficulty). Furthermore, we propose a metric to measure the lexicon usage diversity of a given MWP corpus, and demonstrate that ASDiv is more diverse than existing corpora. Experiments show that our proposed corpus reflects the true capability of MWP solvers more faithfully.
     </details>

18. **Parsing Algebraic Word Problems into Equations** [[pdf]](https://aclanthology.org/Q15-1042) `2015-01-01` (252 cite) (37 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper formalizes the problem of solving multi-sentence algebraic word problems as that of generating and scoring equation trees. We use integer linear programming to generate equation trees and score their likelihood by learning local and global discriminative models. These models are trained on a small set of word problems and their answers, without any manual annotation, in order to choose the equation that best matches the problem text. We refer to the overall system as Alges. We compare Alges with previous work and show that it covers the full gamut of arithmetic operations whereas Hosseini et al. (2014) only handle addition and subtraction. In addition, Alges overcomes the brittleness of the Kushman et al. (2014) approach on single-equation problems, yielding a 15% to 50% reduction in error.
     </details>

19. **Self-Refine: Iterative Refinement with Self-Feedback** [[pdf]](http://arxiv.org/abs/2303.17651) `NeurIPS 2023 Poster` (846 cite) (31 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Self-Refine is introduced, an approach for improving initial outputs from LLMs through iterative feedback and refinement that demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test time using this simple, standalone approach.
     </details>


     <details>
          <summary>Abstract</summary>
          Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides *feedback* for its output and uses it to *refine* itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider. We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by $\sim$20\% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test-time using our simple, standalone approach.
     </details>

20. **ReAct: Synergizing Reasoning and Acting in Language Models** [[pdf]](http://arxiv.org/abs/2210.03629) `ICLR 2023` (1000 cite) (29 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The use of LLMs are explored to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources to gather additional information.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples.
     </details>

21. **Analysing Mathematical Reasoning Abilities of Neural Models** [[pdf]](https://www.semanticscholar.org/paper/afed6dc6900d3b37e528b9086661bba583d60bf6) `ICLR 2019` (360 cite) (29 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper conducts a comprehensive analysis of models from two broad classes of the most powerful sequence-to-sequence architectures and finds notable differences in their ability to resolve mathematical problems and generalize their knowledge.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

22. **Let's Verify Step by Step** [[pdf]](https://arxiv.org/abs/2305.20050) `ICLR 2024 Poster` (344 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work conducts its own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset and shows that active learning significantly improves the efficacy of process supervision.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.
     </details>

23. **Proof Artifact Co-Training for Theorem Proving with Language Models** [[pdf]](https://arxiv.org/abs/2102.06203) `ICLR 2022` `Lean` (98 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PACT is proposed, a general methodology for extracting abundant self-supervised data from kernel-level proof terms for co-training alongside the usual tactic prediction objective and applied to Lean, an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date.
     </details>


     <details>
          <summary>Abstract</summary>
          Labeled data for imitation learning of theorem proving in large libraries of formalized mathematics is scarce as such libraries require years of concentrated effort by human specialists to be built. This is particularly challenging when applying large Transformer language models to tactic prediction, because the scaling of performance with respect to model size is quickly disrupted in the data-scarce, easily-overfitted regime. We propose PACT (Proof Artifact Co-Training), a general methodology for extracting abundant self-supervised data from kernel-level proof terms for joint training alongside the usual tactic prediction objective. We apply this methodology to Lean,an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date. We instrument Lean with a neural theorem prover driven by a Transformer language model and show that PACT improves theorem proving success rate on a held-out suite of test theorems from 32% to 48%.
     </details>

24. **A Goal-Driven Tree-Structured Neural Model for Math Word Problems** [[pdf]](https://www.ijcai.org/proceedings/2019/736) `IJCAI 2019 Natural Language Processing` (194 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A treestructured neural model to generate expression tree in a goal-driven manner is proposed and experimental results on the dataset Math23K have shown that the model outperforms significantly several state-of-the-art models.
     </details>


     <details>
          <summary>Abstract</summary>
          Most existing neural models for math word problems exploit Seq2Seq model to generate solution expressions sequentially from left to right, whose results are far from satisfactory due to the lack of goal-driven mechanism commonly seen in human problem solving. This paper proposes a tree-structured neural model to generate expression tree in a goal-driven manner. Given a math word problem, the model first identifies and encodes its goal to achieve, and then the goal gets decomposed into sub-goals combined by an operator in a top-down recursive way. The whole process is repeated until the goal is simple enough to be realized by a known quantity as leaf node. During the process, two-layer gated-feedforward networks are designed to implement each step of goal decomposition, and a recursive neural network is used to encode fulfilled subtrees into subtree embeddings, which provides a better representation of subtrees than the simple goals of subtrees. Experimental results on the dataset Math23K have shown that our tree-structured model outperforms significantly several state-of-the-art models.
     </details>

25. **WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct** [[pdf]](http://arxiv.org/abs/2308.09583) `ICLR 2025 Submission` (262 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          WizardMath is presented, which enhances the mathematical reasoning abilities of Llama-2, by applying the proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs), such as GPT-4, have shown remarkable performance in natural language processing (NLP) tasks, including challenging mathematical reasoning. However, most existing open-source models are only pre-trained on large-scale internet data and without math-related optimization. In this paper, we present WizardMath, which enhances the mathematical reasoning abilities of Llama-2, by applying our proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math. Through extensive experiments on two mathematical reasoning benchmarks, namely GSM8k and MATH, we reveal the extraordinary capabilities of our model. WizardMath surpasses all other open-source LLMs by a substantial margin. Furthermore, our model even outperforms ChatGPT-3.5, Claude Instant-1, PaLM-2 and Minerva on GSM8k, simultaneously surpasses Text-davinci-002, PaLM-1 and GPT-3 on MATH. More details and model weights are public at https://github.com/nlpxucan/WizardLM and https://huggingface.co/WizardLM.
     </details>

26. **PAL: Program-aided Language Models** [[pdf]](http://arxiv.org/abs/2211.10435) `ICML 2023 Poster` (330 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents Program-Aided Language models (PAL): a novel approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated an impressive ability to perform arithmetic and symbolic reasoning tasks, when provided with a few examples at test time ("few-shot prompting"). Much of this success can be attributed to prompting methods such as "chain-of-thought", which employ LLMs for both understanding the problem description by decomposing it into steps, as well as solving each step of the problem. While LLMs seem to be adept at this sort of step-by-step decomposition, LLMs often make logical and arithmetic mistakes in the solution part, even when the problem is decomposed correctly. In this paper, we present Program-Aided Language models (PAL): a novel approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter. With PAL, decomposing the natural language problem into runnable steps remains the only learning task for the LLM, while solving is delegated to the interpreter. We demonstrate this synergy between a neural LLM and a symbolic interpreter across 13 mathematical, symbolic, and algorithmic reasoning tasks from BIG-Bench Hard and others. In all these natural language reasoning tasks, generating code using an LLM and reasoning using a Python interpreter leads to more accurate results than much larger models. For example, PAL using Codex achieves state-of-the-art few-shot accuracy on GSM8K, surpassing PaLM which uses chain-of-thought by absolute 15% top-1.
     </details>

27. **DeepMath - Deep Sequence Models for Premise Selection** [[pdf]](http://arxiv.org/abs/1606.04442) `NeurIPS 2016` `Mizar` (212 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A two stage approach is proposed that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the effectiveness of neural sequence models for premise selection in automated theorem proving, one of the main bottlenecks in the formalization of mathematics. We propose a two stage approach for this task that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models. To our knowledge, this is the first time deep learning has been applied to theorem proving on a large scale.
     </details>

28. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** [[pdf]](http://arxiv.org/abs/2305.10601) `NeurIPS 2023 Oral` (1000 cite) (25 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices.Our experiments show that ToT significantly enhances language models’ problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4\% of tasks, our method achieved a success rate of 74\%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.
     </details>

29. **Complexity-Based Prompting for Multi-step Reasoning** [[pdf]](http://arxiv.org/abs/2210.00720) `ICLR 2023 Poster` (309 cite) (24 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes complexity-based prompting, a simple and effective example selection scheme for multi-step reasoning that substantially improves multi- step reasoning accuracy and achieves new state-of-the-art (SOTA) performance on three math benchmarks and two BigBenchHard tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the task of prompting large-scale language models to perform multi-step reasoning. Existing work shows that when prompted with a chain of thoughts (CoT), sequences of short sentences describing intermediate reasoning steps towards a final answer, large language models can generate new reasoning chains and predict answers for new inputs. A central question is which reasoning examples make the most effective prompts. In this work, we propose complexity-based prompting, a simple and effective example selection scheme for multi-step reasoning. We show that prompts with higher reasoning complexity, i.e., chains with more reasoning steps, achieve substantially better performance on math word reasoning tasks over strong baselines. We further extend our complexity-based criteria from prompting (selecting inputs) to decoding (selecting outputs), where we sample multiple reasoning chains from the model, then choose the majorityof generated answers from complex reasoning chains (over simple chains). When used to prompt GPT-3, our approach substantially improves multi-step reasoning accuracy, with an 8.6% absolute improvement on GSM8K, and 6.4% on MathQA. Compared with existing example selection schemes like manual tuning or retrieval-based selection, selection based on reasoning complexity is intuitive, easy to implement, and annotation-efficient. Further results demonstrate the robustness of performance gains from complex prompts under format perturbation and distribution shift.
     </details>

30. **Toolformer: Language Models Can Teach Themselves to Use Tools** [[pdf]](http://arxiv.org/abs/2302.04761) `NeurIPS 2023 Oral` (1000 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction, which achieves substantially improved zero-shot performance across a variety of downstream tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller specialized models excel. In this paper, we show that LMs can teach themselves to *use external tools* via simple APIs and achieve the best of both worlds. We introduce *Toolformer*, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q&A system, a search engine, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.
     </details>

31. **miniF2F: a cross-system benchmark for formal Olympiad-level mathematics** [[pdf]](https://arxiv.org/abs/2109.00110) `ICLR 2022` `Lean, Isabelle, HOL Light, MetaMath` (84 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The miniF2F benchmark currently targets Metamath, Lean, Isabelle, and HOL Light and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad, as well as material from high-school and undergraduate mathematics courses.
     </details>


     <details>
          <summary>Abstract</summary>
          We present $\textsf{miniF2F}$, a dataset of formal Olympiad-level mathematics problems statements intended to provide a unified cross-system benchmark for neural theorem proving. The $\textsf{miniF2F}$ benchmark currently targets Metamath, Lean, Isabelle (partially) and HOL Light (partially) and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad (IMO), as well as material from high-school and undergraduate mathematics courses. We report baseline results using GPT-f, a neural theorem prover based on GPT-3 and provide an analysis of its performance. We intend for $\textsf{miniF2F}$ to be a community-driven effort and hope that our benchmark will help spur advances in neural theorem proving.
     </details>

32. **Measuring Massive Multitask Language Understanding** [[pdf]](http://arxiv.org/abs/2009.03300) `ICLR 2021` (1000 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          While most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average, however, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.
     </details>

33. **Deep Learning for Symbolic Mathematics** [[pdf]](http://arxiv.org/abs/1912.01412) `ICLR 2020 Spotlight` (357 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that neural networks can be surprisingly good at more elaborated tasks in mathematics, such as symbolic integration and solving differential equations, and a syntax for representing these mathematical problems, and methods for generating large datasets that can be used to train sequence-to-sequence models.
     </details>


     <details>
          <summary>Abstract</summary>
          Neural networks have a reputation for being better at solving statistical or approximate problems than at performing calculations or working with symbolic data. In this paper, we show that they can be surprisingly good at more elaborated tasks in mathematics, such as symbolic integration and solving differential equations. We propose a syntax for representing mathematical problems, and methods for generating large datasets that can be used to train sequence-to-sequence models. We achieve results that outperform commercial Computer Algebra Systems such as Matlab or Mathematica.
     </details>

34. **MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning** [[pdf]](http://arxiv.org/abs/2309.05653) `ICLR 2024` (232 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%, and underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce MAmmoTH, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MAmmoTH models are trained on MathInstruct, our meticulously curated instruction tuning dataset. MathInstruct is compiled from 13 math datasets with intermediate rationales, six of which have rationales newly curated by us. It presents a unique hybrid of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT not only unleashes the potential of tool use but also allows different thought processes for different math problems. As a result, the MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%. Remarkably, our MAmmoTH-7B model reaches 33% on MATH (a competition-level dataset), which exceeds the best open-source 7B model (WizardMath) by 23%, and the MAmmoTH-34B model achieves 44% accuracy on MATH, even surpassing GPT-4's CoT result. Our work underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.
     </details>

35. **Template-Based Math Word Problem Solvers with Recursive Neural Networks** [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/4697) `AAAI 2019 Natural Language Processing` (134 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper designs a recursive neural network to encode the quantity with Bi-LSTM and self attention, and infer the unknown operator nodes in a bottom-up manner, and establishes the superiority of the new framework as it improves the accuracy by a wide margin in two of the largest datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          The design of automatic solvers to arithmetic math word problems has attracted considerable attention in recent years and a large number of datasets and methods have been published. Among them, Math23K is the largest data corpus that is very helpful to evaluate the generality and robustness of a proposed solution. The best performer in Math23K is a seq2seq model based on LSTM to generate the math expression. However, the model suffers from performance degradation in large space of target expressions. In this paper, we propose a template-based solution based on recursive neural network for math expression construction. More specifically, we first apply a seq2seq model to predict a tree-structure template, with inferred numbers as leaf nodes and unknown operators as inner nodes. Then, we design a recursive neural network to encode the quantity with Bi-LSTM and self attention, and infer the unknown operator nodes in a bottom-up manner. The experimental results clearly establish the superiority of our new framework as we improve the accuracy by a wide margin in two of the largest datasets, i.e., from 58.1% to 66.9% in Math23K and from 62.8% to 66.8% in MAWPS.
     </details>

36. **Learning to Prove Theorems via Interacting with Proof Assistants** [[pdf]](https://www.semanticscholar.org/paper/a596f03145285cd05a6ca57a4e25418b23b24976) `ICML 2019` `Coq` (117 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ASTactic, a deep learning-based model that generates tactics as programs in the form of abstract syntax trees (ASTs) can generate effective tactics and can be used to prove new theorems not previously provable by automated methods.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

37. **Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them** [[pdf]](https://arxiv.org/abs/2210.09261v1) `ACL 2023 Findings` (641 cite) (21 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work finds that applying chain-of-thought (CoT) prompting to BBH tasks enables PaLM to surpass the average human-rater performance on 10 of the 23 tasks, and Codex to surpass it on 17 of the23 tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models? In this work, we focus on a suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH). These are the tasks for which prior language model evaluations did not outperform the average human-rater. We find that applying chain-of-thought (CoT) prompting to BBH tasks enables PaLM to surpass the average human-rater performance on 10 of the 23 tasks, and Codex (code-davinci-002) to surpass the average human-rater performance on 17 of the 23 tasks. Since many tasks in BBH require multi-step reasoning, few-shot prompting without CoT, as done in the BIG-Bench evaluations (Srivastava et al., 2022), substantially underestimates the best performance and capabilities of language models, which is better captured via CoT prompting. As further analysis, we explore the interaction between CoT and model scale on BBH, finding that CoT enables emergent task performance on several BBH tasks with otherwise flat scaling curves.
     </details>

38. **How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation** [[pdf]](https://aclanthology.org/P16-1084) `ACL 2016 Long Papers` (130 cite) (21 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A large-scale dataset is built which is more than 9 times the size of previous ones, and contains many more problem types, and semi-automatically obtained from community question-answering web pages.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

39. **STaR: Bootstrapping Reasoning With Reasoning** [[pdf]](http://arxiv.org/abs/2203.14465) `NeurIPS 2022 Poster` (260 cite) (20 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A technique to iteratively leverage a small number of rationale examples and a large dataset without rationales to bootstrap the ability to perform successively more complex reasoning, called STaR, which lets a model improve itself by learning from its own generated reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Generating step-by-step "chain-of-thought" rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering. However, inducing language model rationale generation currently requires either constructing massive rationale datasets or sacrificing accuracy by using only few-shot inference. We propose a technique to iteratively leverage a small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning. This technique, the "Self-Taught Reasoner" (STaR), relies on a simple loop: generate rationales to answer many questions, prompted with a few rationale examples; if the generated answers are wrong, try again to generate a rationale given the correct answer; fine-tune on all the rationales that ultimately yielded correct answers; repeat. We show that STaR significantly improves performance on multiple datasets compared to a model fine-tuned to directly predict final answers, and performs comparably to fine-tuning a 30$\times$ larger state-of-the-art language model on CommensenseQA. Thus, STaR lets a model improve itself by learning from its own generated reasoning.
     </details>

40. **Graph-to-Tree Learning for Solving Math Word Problems** [[pdf]](https://aclanthology.org/2020.acl-main.362) `ACL 2020 Main` (166 cite) (20 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph2Tree is proposed, a novel deep learning architecture that combines the merits of the graph-based encoder and tree-based decoder to generate better solution expressions for math word problem (MWP) solution expressions.
     </details>


     <details>
          <summary>Abstract</summary>
          While the recent tree-based neural models have demonstrated promising results in generating solution expression for the math word problem (MWP), most of these models do not capture the relationships and order information among the quantities well. This results in poor quantity representations and incorrect solution expressions. In this paper, we propose Graph2Tree, a novel deep learning architecture that combines the merits of the graph-based encoder and tree-based decoder to generate better solution expressions. Included in our Graph2Tree framework are two graphs, namely the Quantity Cell Graph and Quantity Comparison Graph, which are designed to address limitations of existing methods by effectively representing the relationships and order information among the quantities in MWPs. We conduct extensive experiments on two available datasets. Our experiment results show that Graph2Tree outperforms the state-of-the-art baselines on two benchmark datasets significantly. We also discuss case studies and empirically examine Graph2Tree’s effectiveness in translating the MWP text into solution expressions.
     </details>

41. **Translating a Math Word Problem to an Expression Tree** [[pdf]](https://www.semanticscholar.org/paper/Translating-a-Math-Word-Problem-to-an-Expression-Wang-Wang/6605bba6e0caabda06b090d67698a5683eba4dfa) `2018-11-14` (144 cite) (20 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By considering the uniqueness of expression tree, an equation normalization method to normalize the duplicated equations is proposed and it is shown that the ensemble model with equationnormalization significantly outperforms the previous state-of-the-art methods.
     </details>


     <details>
          <summary>Abstract</summary>
          Sequence-to-sequence (SEQ2SEQ) models have been successfully applied to automatic math word problem solving. Despite its simplicity, a drawback still remains: a math word problem can be correctly solved by more than one equations. This non-deterministic transduction harms the performance of maximum likelihood estimation. In this paper, by considering the uniqueness of expression tree, we propose an equation normalization method to normalize the duplicated equations. Moreover, we analyze the performance of three popular SEQ2SEQ models on the math word problem solving. We find that each model has its own specialty in solving problems, consequently an ensemble model is then proposed to combine their advantages. Experiments on dataset Math23K show that the ensemble model with equation normalization significantly outperforms the previous state-of-the-art methods.
     </details>

42. **Automatically Solving Number Word Problems by Semantic Parsing and Reasoning** [[pdf]](http://aclweb.org/anthology/D15-1135) `2015-01-01` (154 cite) (19 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new meaning representation language is designed to bridge natural language text and math expressions and a CFG parser is implemented based on 9,600 semi-automatically created grammar rules.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a semantic parsing and reasoning approach to automatically solving math word problems. A new meaning representation language is designed to bridge natural language text and math expressions. A CFG parser is implemented based on 9,600 semi-automatically created grammar rules. We conduct experiments on a test set of over 1,500 number word problems (i.e., verbally expressed number problems) and yield 95.4% precision and 60.2% recall.
     </details>

43. **Decomposed Prompting: A Modular Approach for Solving Complex Tasks** [[pdf]](https://openreview.net/forum?id=_nGgzQjzaRy) `ICLR 2023 Poster` (284 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that the flexibility and modularity of Decomposed Prompting allows it to outperform prior work on few-shot prompting using GPT3 and to incorporate a symbolic information retrieval within the decomposition framework, leading to improved performance on both tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Few-shot prompting is a surprisingly powerful way to use Large Language Models (LLMs) to solve various tasks. However, this approach struggles as the task complexity increases or when the individual reasoning steps of the task themselves are hard to learn, especially when embedded in more complex tasks. To address this, we propose Decomposed Prompting, a new approach to solve complex tasks by decomposing them (via prompting) into simpler sub-tasks that can be delegated to a library of prompting-based LLMs dedicated to these sub-tasks. This modular structure allows each prompt to be optimized for its specific sub-task, further decomposed if necessary, and even easily replaced with more effective prompts, trained models, or symbolic functions if desired.We show that the flexibility and modularity of Decomposed Prompting allows it to outperform prior work on few-shot prompting using GPT3. On symbolic reasoning tasks, we can further decompose sub-tasks that are hard for LLMs into even simpler solvable sub-tasks. When the complexity comes from the input length, we can recursively decompose the task into the same task but with smaller inputs. We also evaluate our approach on textual multi-step reasoning tasks: on long-context multi-hop QA task, we can more effectively teach the sub-tasks via our separate sub-tasks prompts; and on open-domain multi-hop QA, we can incorporate a symbolic information retrieval within our decomposition framework, leading to improved performance on both tasks. Datasets, Code and Prompts available at https://github.com/allenai/DecomP.
     </details>

44. **HyperTree Proof Search for Neural Theorem Proving** [[pdf]](http://arxiv.org/abs/2205.11491) `NeurIPS 2022` `Lean, MetaMath` (84 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose an online training procedure for a transformer-based automated theorem prover. Our approach leverages a new search algorithm, HyperTree Proof Search (HTPS), that learns from previous proof searches through online training, allowing it to generalize to domains far from the training distribution. We report detailed ablations of our pipeline’s main components by studying performance on three environments of increasing complexity. In particular, we show that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f. Online training on these unproved theorems increases accuracy to 82.6%. With a similar computational budget, we improve the state of the art on the Lean-based miniF2F-curriculum dataset from 31% to 42% proving accuracy.
     </details>

45. **LILA: A Unified Benchmark for Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2210.17517) `EMNLP 2022 Main` (103 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that multi-tasking leads to significant improvements (average relative improvement of 21.83% F1 score vs. single-task models), indicating the room for improvement in general mathematical reasoning and understanding.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning skills are essential for general-purpose intelligentsystems to perform tasks from grocery shopping to climate modeling. Towards evaluating and improving AI systems in this domain, we proposeLILA, a unified mathematical reasoning benchmark consisting of 23 diversetasks along four dimensions:(i) mathematical abilities e.g., arithmetic, calculus (ii) language format e.g., question-answering, fill-in-the-blanks (iii) language diversity e.g., no language, simple language (iv) external knowledge e.g., commonsense, physics. We construct our benchmark by extending 20 datasets benchmark by collecting task instructions and solutions in the form of Python programs,thereby obtaining explainable solutions in addition to the correct answer. We additionally introduce two evaluation datasets to measure out-of-distribution performance and robustness to language perturbation. Finally, we introduce BHASKARA,a general-purpose mathematical reasoning model trained on LILA. Importantly, we find that multi-tasking leads to significant improvements (average relative improvement of 21.83% F1 score vs. single-task models),while the best performing model only obtains 60.40%,indicating the room for improvement in general mathematical reasoning and understanding.
     </details>

46. **Generate & Rank: A Multi-task Framework for Math Word Problems** [[pdf]](http://arxiv.org/abs/2109.03034) `EMNLP 2021 Findings` (0 cite) (18 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math word problem (MWP) is a challenging and critical task in natural language processing. Many recent studies formalize MWP as a generation task and have adopted sequence-to-sequence models to transform problem descriptions to mathematical expressions. However, mathematical expressions are prone to minor mistakes while the generation objective does not explicitly handle such mistakes. To address this limitation, we devise a new ranking task for MWP and propose Generate & Rank, a multi-task framework based on a generative pre-trained language model. By joint training with generation and ranking, the model learns from its own mistakes and is able to distinguish between correct and incorrect expressions. Meanwhile, we perform tree-based disturbance specially designed for MWP and an online update to boost the ranker. We demonstrate the effectiveness of our proposed method on the benchmark and the results show that our method consistently outperforms baselines in all datasets. Particularly, in the classical Math23k, our method is 7% (78.4% to 85.4%) higher than the state-of-the-art. Code could be found at https://github.com/huawei-noah/noah-research.
     </details>

47. **Mapping to Declarative Knowledge for Word Problem Solving** [[pdf]](https://aclanthology.org/Q18-1012) `2018-01-01` (89 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Declarative rules which govern the translation of natural language description of these concepts to math expressions are developed, and a framework for incorporating such declarative knowledge into word problem solving is presented.
     </details>


     <details>
          <summary>Abstract</summary>
          Math word problems form a natural abstraction to a range of quantitative reasoning problems, such as understanding financial news, sports results, and casualties of war. Solving such problems requires the understanding of several mathematical concepts such as dimensional analysis, subset relationships, etc. In this paper, we develop declarative rules which govern the translation of natural language description of these concepts to math expressions. We then present a framework for incorporating such declarative knowledge into word problem solving. Our method learns to map arithmetic word problem text to math expressions, by learning to select the relevant declarative knowledge for each operation of the solution expression. This provides a way to handle multiple concepts in the same problem while, at the same time, supporting interpretability of the answer expression. Our method models the mapping to declarative knowledge as a latent variable, thus removing the need for expensive annotations. Experimental evaluation suggests that our domain knowledge based solver outperforms all other systems, and that it generalizes better in the realistic case where the training data it is exposed to is biased in a different way than the test data.
     </details>

48. **Deep Network Guided Proof Search** [[pdf]](https://www.semanticscholar.org/paper/e77e00286a63a32dafb9629cd79f6a77bddb1941) `2017-01-01` (148 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental evidence is given that with a hybrid, two-phase approach, deep learning based guidance can significantly reduce the average number of proof search steps while increasing the number of theorems proved.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

49. **Reasoning about Quantities in Natural Language** [[pdf]](https://direct.mit.edu/tacl/article-abstract/doi/10.1162/tacl_a_00118/43260) `2015-01-01` (149 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A computational approach is developed which is shown to successfully recognize and normalize textual expressions of quantities and is used to further develop algorithms to assist reasoning in the context of the aforementioned tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

50. **Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification** [[pdf]](http://arxiv.org/abs/2308.07921) `ICLR 2024 Poster` (106 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The effect of code on enhancing LLMs' reasoning capability by introducing different constraints on the Code Usage Frequency of GPT-4 Code Interpreter is explored, and a novel and effective prompting method, explicit \uline{c}ode-based \ULine{s}elf-\uline {v}erification~(CSV), is proposed to further boost the mathematical reasoning potential of GPN.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent progress in large language models (LLMs) like GPT-4 and PaLM-2 has brought significant advancements in addressing math reasoning problems. In particular, OpenAI's latest version of GPT-4, known as GPT-4 Code Interpreter, shows remarkable performance on challenging math datasets. In this paper, we explore the effect of code on enhancing LLMs' reasoning capability by introducing different constraints on the Code Usage Frequency of GPT-4 Code Interpreter. We found that its success can be largely attributed to its powerful skills in generating and executing code, evaluating the output of code execution, and rectifying its solution when receiving unreasonable outputs. Based on this insight, we propose a novel and effective prompting method, explicit $\underline{\text{c}}$ode-based $\underline{\text{s}}$elf-$\underline{\text{v}}$erification (CSV), to further boost the mathematical reasoning potential of GPT-4 Code Interpreter. This method employs a zero-shot prompt on GPT-4 Code Interpreter to encourage it to use code to self-verify its answers. In instances where the verification state registers as "False", the model shall automatically amend its solution, analogous to our approach of rectifying errors during a mathematics examination. Furthermore, we recognize that the states of the verification result indicate the confidence of a solution, which can improve the effectiveness of majority voting. With GPT-4 Code Interpreter and CSV, we achieve an impressive zero-shot accuracy on MATH dataset $\textbf{(53.9}$% → $\textbf{84.3}$%$\textbf{)}$.
     </details>

51. **Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2209.14610) `ICLR 2023` (181 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach is proposed, PromptPG, which utilizes policy gradient to learn to select in-context examples from a small amount of training data and then constructs the corresponding prompt for the test example, which verifies its effectiveness in selecting in- context examples.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning, a core ability of human intelligence, presents unique challenges for machines in abstract thinking and logical reasoning. Recent large pre-trained language models such as GPT-3 have achieved remarkable progress on mathematical reasoning tasks written in text form, such as math word problems (MWP). However, it is unknown if the models can handle more complex problems that involve math reasoning over heterogeneous information, such as tabular data. To fill the gap, we present Tabular Math Word Problems (TabMWP), a new dataset containing 38,431 open-domain grade-level problems that require mathematical reasoning on both textual and tabular data. Each question in TabMWP is aligned with a tabular context, which is presented as an image, semi-structured text, and a structured table. There are two types of questions: free-text and multi-choice, and each problem is annotated with gold solutions to reveal the multi-step reasoning process. We evaluate different pre-trained models on TabMWP, including the GPT-3 model in a few-shot setting. As earlier studies suggest, since few-shot GPT-3 relies on the selection of in-context examples, its performance is unstable and can degrade to near chance. The unstable issue is more severe when handling complex problems like TabMWP. To mitigate this, we further propose a novel approach, PromptPG, which utilizes policy gradient to learn to select in-context examples from a small amount of training data and then constructs the corresponding prompt for the test example. Experimental results show that our method outperforms the best baseline by 5.31% on the accuracy metric and reduces the prediction variance significantly compared to random selection, which verifies its effectiveness in selecting in-context examples. The data and code are available at https://promptpg.github.io.
     </details>

52. **Galactica: A Large Language Model for Science** [[pdf]](http://arxiv.org/abs/2211.09085) `2022-11-16` (580 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Galactica is introduced: a large language model that can store, combine and reason about scientific knowledge, and sets a new state-of-the-art on downstream tasks such as PubMedQA and MedMCQA dev of 77.6% and 52.9%.
     </details>


     <details>
          <summary>Abstract</summary>
          Information overload is a major obstacle to scientific progress. The explosive growth in scientific literature and data has made it ever harder to discover useful insights in a large mass of information. Today scientific knowledge is accessed through search engines, but they are unable to organize scientific knowledge alone. In this paper we introduce Galactica: a large language model that can store, combine and reason about scientific knowledge. We train on a large scientific corpus of papers, reference material, knowledge bases and many other sources. We outperform existing models on a range of scientific tasks. On technical knowledge probes such as LaTeX equations, Galactica outperforms the latest GPT-3 by 68.2% versus 49.0%. Galactica also performs well on reasoning, outperforming Chinchilla on mathematical MMLU by 41.3% to 35.7%, and PaLM 540B on MATH with a score of 20.4% versus 8.8%. It also sets a new state-of-the-art on downstream tasks such as PubMedQA and MedMCQA dev of 77.6% and 52.9%. And despite not being trained on a general corpus, Galactica outperforms BLOOM and OPT-175B on BIG-bench. We believe these results demonstrate the potential for language models as a new interface for science. We open source the model for the benefit of the scientific community.
     </details>

53. **Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs** [[pdf]](http://arxiv.org/abs/2210.12283) `ICLR 2023` `Isabelle` (99 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems, is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          The formalization of existing mathematical proofs is a notoriously difficult process. Despite decades of research on automation and proof assistants, writing formal proofs remains arduous and only accessible to a few experts. While previous studies to automate formalization focused on powerful search algorithms, no attempts were made to take advantage of available informal proofs. In this work, we introduce Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems. We investigate two relevant setups where informal proofs are either written by humans or generated by a language model. Our experiments and ablation studies show that large language models are able to produce well-structured formal sketches that follow the same reasoning steps as the informal proofs. Guiding an automated prover with these sketches enhances its performance from $20.9\%$ to $39.3\%$ on a collection of mathematical competition problems.
     </details>

54. **Formal Mathematics Statement Curriculum Learning** [[pdf]](http://arxiv.org/abs/2202.01344) `ICLR 2023` `Lean` (0 cite) (17 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We explore the use of expert iteration in the context of language modeling applied to formal mathematics. We show that at same compute budget, expert iteration, by which we mean proof search interleaved with learning, dramatically outperforms proof search only. We also observe that when applied to a collection of formal statements of sufficiently varied difficulty, expert iteration is capable of finding and solving a curriculum of increasingly difficult problems, without the need for associated ground-truth proofs. Finally, by applying this expert iteration to a manually curated set of problem statements, we surpass previous state-of-the-art on the miniF2F benchmark, automatically solving multiple challenging problems drawn from high school olympiads.
     </details>

55. **INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving** [[pdf]](https://arxiv.org/abs/2007.02924) `ICLR 2021` (49 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces INT, an INequality Theorem proving benchmark, specifically designed to test agents' generalization ability, and evaluates the same agents augmented with Monte Carlo Tree Search at test time, and shows that MCTS can help to prove new theorems.
     </details>


     <details>
          <summary>Abstract</summary>
          In learning-assisted theorem proving, one of the most critical challenges is to generalize to theorems unlike those seen at training time. In this paper, we introduce INT, an INequality Theorem proving benchmark designed to test agents’ generalization ability. INT is based on a theorem generator, which provides theoretically infinite data and allows us to measure 6 different types of generalization, each reflecting a distinct challenge, characteristic of automated theorem proving. In addition, provides a fast theorem proving environment with sequence-based and graph-based interfaces, conducive to performing learning-based research. We introduce base-lines with architectures including transformers and graph neural networks (GNNs)for INT. Using INT, we find that transformer-based agents achieve stronger test performance for most of the generalization tasks, despite having much larger out-of-distribution generalization gaps than GNNs. We further find that the addition of Monte Carlo Tree Search (MCTS) at test time helps to prove new theorems.
     </details>

56. **IsarStep: a Benchmark for High-level Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2006.09265) `ICLR 2021` `Isabelle` (52 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A benchmark for high-level mathematical reasoning is presented and the reasoning capabilities of neural sequence-to-sequence models are studied and a hierarchical transformer is designed that outperforms the transformer baseline.
     </details>


     <details>
          <summary>Abstract</summary>
          A well-defined benchmark is essential for measuring and accelerating research progress of machine learning models. In this paper, we present a benchmark for high-level mathematical reasoning and study the reasoning capabilities of neural sequence-to-sequence models. We build a non-synthetic dataset from the largest repository of proofs written by human experts in a theorem prover. The dataset has a broad coverage of undergraduate and research-level mathematical and computer science theorems. In our defined task, a model is required to fill in a missing intermediate proposition given surrounding proofs. This task provides a starting point for the long-term goal of having machines generate human-readable proofs automatically. Our experiments and analysis reveal that while the task is challenging, neural models can capture non-trivial mathematical reasoning. We further design a hierarchical transformer that outperforms the transformer baseline.
     </details>

57. **Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems** [[pdf]](http://aclweb.org/anthology/N19-1272) `NAACL 2019 Main` (100 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed neural math solver is based on an encoder-decoder framework, where the encoder is designed to understand the semantics of problems, and the decoder focuses on tracking semantic meanings of the generated symbols and then deciding which symbol to generate next.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving math word problems is a challenging task that requires accurate natural language understanding to bridge natural language texts and math expressions. Motivated by the intuition about how human generates the equations given the problem texts, this paper presents a neural approach to automatically solve math word problems by operating symbols according to their semantic meanings in texts. This paper views the process of generating equation as a bridge between the semantic world and the symbolic world, where the proposed neural math solver is based on an encoder-decoder framework. In the proposed model, the encoder is designed to understand the semantics of problems, and the decoder focuses on tracking semantic meanings of the generated symbols and then deciding which symbol to generate next. The preliminary experiments are conducted in a dataset Math23K, and our model significantly outperforms both the state-of-the-art single model and the best non-retrieval-based model over about 10% accuracy, demonstrating the effectiveness of bridging the symbolic and semantic worlds from math word problems.
     </details>

58. **REFINER: Reasoning Feedback on Intermediate Representations** [[pdf]](http://arxiv.org/abs/2304.01904) `2024-02-04` (108 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          REFINER is a framework for finetuning LMs to explicitly generate intermediate reasoning steps while interacting with a critic model that provides automated feedback on the reasoning that provides structured feedback that the reasoning LM uses to iteratively improve its intermediate arguments.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models (LMs) have recently shown remarkable performance on reasoning tasks by explicitly generating intermediate inferences, e.g., chain-of-thought prompting. However, these intermediate inference steps may be inappropriate deductions from the initial context and lead to incorrect final predictions. Here we introduce REFINER, a framework for finetuning LMs to explicitly generate intermediate reasoning steps while interacting with a critic model that provides automated feedback on the reasoning. Specifically, the critic provides structured feedback that the reasoning LM uses to iteratively improve its intermediate arguments. Empirical evaluations of REFINER on three diverse reasoning tasks show significant improvements over baseline LMs of comparable scale. Furthermore, when using GPT-3.5 or ChatGPT as the reasoner, the trained critic significantly improves reasoning without finetuning the reasoner. Finally, our critic model is trained without expensive human-in-the-loop data but can be substituted with humans at inference time.
     </details>

59. **A Survey of Deep Learning for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2212.10535) `ACL 2023` (99 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This survey paper reviews the key tasks, datasets, and methods at the intersection of mathematical reasoning and deep learning over the past decade, and evaluates existing benchmarks and methods and discusses future research directions in this domain.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is a fundamental aspect of human intelligence and is applicable in various fields, including science, engineering, finance, and everyday life. The development of artificial intelligence (AI) systems capable of solving math problems and proving theorems in language has garnered significant interest in the fields of machine learning and natural language processing. For example, mathematics serves as a testbed for aspects of reasoning that are challenging for powerful deep learning models, driving new algorithmic and modeling advances. On the other hand, recent advances in large-scale neural language models have opened up new benchmarks and opportunities to use deep learning for mathematical reasoning. In this survey paper, we review the key tasks, datasets, and methods at the intersection of mathematical reasoning and deep learning over the past decade. We also evaluate existing benchmarks and methods, and discuss future research directions in this domain.
     </details>

60. **Autoformalization with Large Language Models** [[pdf]](https://arxiv.org/abs/2205.12615) `NeurIPS 2022` `Lean` (108 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown large language models provide new prospects towards the long-term goal of autoformalization, and the surprising observation that LLMs can correctly translate a significant portion of mathematical competition problems perfectly to formal specifications in Isabelle/HOL.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the process of automatically translating from natural language mathematics to formal specifications and proofs. A successful autoformalization system could advance the fields of formal verification, program synthesis, and artificial intelligence.While the long-term goal of autoformalization seemed elusive for a long time, we show large language models provide new prospects towards this goal. We make the surprising observation that LLMs can correctly translate a significant portion ($25.3\%$) of mathematical competition problems perfectly to formal specifications in Isabelle/HOL. We demonstrate the usefulness of this process by improving a previously introduced neural theorem prover via training on these autoformalized theorems. Our methodology results in a new state-of-the-art result on the MiniF2F theorem proving benchmark, improving the proof rate from~$29.6\%$ to~$35.2\%$.
     </details>

61. **NaturalProofs: Mathematical Theorem Proving in Natural Language** [[pdf]](https://arxiv.org/abs/2104.01112) `NeurIPS 2021` (51 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          NaturalProofs is developed, a multi-domain corpus of mathematical statements and their proofs, written in natural mathematical language that unifies broad coverage, deep coverage, and low-resource mathematical sources, allowing for evaluating both in-distribution and zero-shot generalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Understanding and creating mathematics using natural mathematical language - the mixture of symbolic and natural language used by humans - is a challenging and important problem for driving progress in machine learning. As a step in this direction, we develop NaturalProofs, a multi-domain corpus of mathematical statements and their proofs, written in natural mathematical language. NaturalProofs unifies broad coverage, deep coverage, and low-resource mathematical sources, allowing for evaluating both in-distribution and zero-shot generalization. Using NaturalProofs, we benchmark strong neural methods on mathematical reference retrieval and generation tasks which test a system's ability to determine key results that appear in a proof. Large-scale sequence models show promise compared to classical information retrieval methods, yet their performance and out-of-domain generalization leave substantial room for improvement. NaturalProofs opens many avenues for research on challenging mathematical tasks.
     </details>

62. **GamePad: A Learning Environment for Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/87c425f23bcac2f082968abda64a971f91522f73) `ICLR 2019` `Coq` (97 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A system called GamePad is introduced that can be used to explore the application of machine learning methods to theorem proving in the Coq proof assistant and addresses position evaluation and tactic prediction tasks, which arise naturally in tactic-based theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

63. **Large Language Models Are Reasoning Teachers** [[pdf]](https://aclanthology.org/2023.acl-long.830) `ACL 2023 Long Papers` (232 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper uses very large models as reasoning teachers to enable complex reasoning in smaller models and reduce model size requirements by several orders of magnitude, and proposes Fine-tune-CoT, a method that generates reasoning samples from very large teacher models to fine-tunes smaller models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent works have shown that chain-of-thought (CoT) prompting can elicit language models to solve complex reasoning tasks, step-by-step. However, prompt-based CoT methods are dependent on very large models such as GPT-3 175B which are prohibitive to deploy at scale. In this paper, we use these large models as reasoning teachers to enable complex reasoning in smaller models and reduce model size requirements by several orders of magnitude. We propose Fine-tune-CoT, a method that generates reasoning samples from very large teacher models to fine-tune smaller models. We evaluate our method on a wide range of public models and complex tasks. We find that Fine-tune-CoT enables substantial reasoning capability in small models, far outperforming prompt-based baselines and even the teacher model in many tasks. Additionally, we extend our method by leveraging the teacher model’s ability to generate multiple distinct rationales for each original sample. Enriching the fine-tuning data with such diverse reasoning results in a substantial performance boost across datasets, even for very small models. We conduct ablations and sample studies to understand the emergence of reasoning capabilities of student models. Our code implementation and data are available at https://github.com/itsnamgyu/reasoning-teacher.
     </details>

64. **Teaching Small Language Models to Reason** [[pdf]](https://aclanthology.org/2023.acl-short.151) `ACL 2023 Short Papers` (175 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper finetune a student model on the chain of thought outputs generated by a larger teacher model, and shows that the proposed method improves task performance across arithmetic, commonsense and symbolic reasoning datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain of thought prompting successfully improves the reasoning capabilities of large language models, achieving state of the art results on a range of datasets. However, these reasoning capabilities only appear to emerge in models with at least tens of billions of parameters. In this paper, we explore the transfer of such reasoning capabilities to smaller models via knowledge distillation, also investigating model and dataset size trade-off. Specifically, we finetune a student model on the chain of thought outputs generated by a larger teacher model. Our experiments show that the proposed method improves task performance across arithmetic, commonsense and symbolic reasoning datasets. For example, the accuracy of T5 XXL on GSM8K improves from 8.11% to 21.99% and 18.42% when finetuned on PaLM 540B and GPT-3 175B generated chains of thought, respectively.
     </details>

65. **Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers** [[pdf]](http://arxiv.org/abs/2205.10893) `NeurIPS 2022` `Isabelle` (65 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Thor is introduced, a framework integrating language models and automated theorem provers to overcome the difficulty of selecting useful premises from a large library to unlock the proof of a given conjecture.
     </details>


     <details>
          <summary>Abstract</summary>
          In theorem proving, the task of selecting useful premises from a large library to unlock the proof of a given conjecture is crucially important. This presents a challenge for all theorem provers, especially the ones based on language models, due to their relative inability to reason over huge volumes of premises in text form. This paper introduces Thor, a framework integrating language models and automated theorem provers to overcome this difficulty. In Thor, a class of methods called hammers that leverage the power of automated theorem provers are used for premise selection, while all other tasks are designated to language models. Thor increases a language model's success rate on the PISA dataset from $39\%$ to $57\%$, while solving $8.2\%$ of problems neither language models nor automated theorem provers are able to solve on their own. Furthermore, with a significantly smaller computational budget, Thor can achieve a success rate on the MiniF2F dataset that is on par with the best existing methods. Thor can be instantiated for the majority of popular interactive theorem provers via a straightforward protocol we provide.
     </details>

66. **Mathematical Reasoning via Self-supervised Skip-tree Training** [[pdf]](https://arxiv.org/abs/2006.04757) `ICLR 2021` (53 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that models trained on the skip-tree task show surprisingly strong mathematical reasoning abilities, and outperform modelstrained on standard skip-sequence tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          We demonstrate that self-supervised language modeling applied to mathematical formulas enables logical reasoning. To measure the logical reasoning abilities of language models, we formulate several evaluation (downstream) tasks, such as inferring types, suggesting missing assumptions and completing equalities. For training language models for formal mathematics, we propose a novel skip-tree task. We find that models trained on the skip-tree task show surprisingly strong mathematical reasoning abilities, and outperform models trained on standard skip-sequence tasks. We also analyze the models' ability to formulate new conjectures by measuring how often the predictions are provable and useful in other proofs.
     </details>

67. **Learning to Reason in Large Theories without Imitation** [[pdf]](https://arxiv.org/abs/1905.10501) `2019-01-01` (38 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper suggests an exploration mechanism that mixes in additional premises selected by a tf-idf (term frequency-inverse document frequency) based lookup in a deep reinforcement learning scenario that outperforms provers that are trained only on human proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we demonstrate how to do automated theorem proving in the presence of a large knowledge base of potential premises without learning from human proofs. We suggest an exploration mechanism that mixes in additional premises selected by a tf-idf (term frequency-inverse document frequency) based lookup in a deep reinforcement learning scenario. This helps with exploring and learning which premises are relevant for proving a new theorem. Our experiments show that the theorem prover trained with this exploration mechanism outperforms provers that are trained only on human proofs. It approaches the performance of a prover trained by a combination of imitation and reinforcement learning. We perform multiple experiments to understand the importance of the underlying assumptions that make our exploration approach work, thus explaining our design choices.
     </details>

68. **Holophrasm: a neural Automated Theorem Prover for higher-order logic** [[pdf]](http://arxiv.org/abs/1608.02644) `2016-08-09` (44 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Holophrasm exploits the formalism of the Metamath language and explores partial proof trees using a neural-network-augmented bandit algorithm and a sequence-to-sequence model for action enumeration.
     </details>


     <details>
          <summary>Abstract</summary>
          I propose a system for Automated Theorem Proving in higher order logic using deep learning and eschewing hand-constructed features. Holophrasm exploits the formalism of the Metamath language and explores partial proof trees using a neural-network-augmented bandit algorithm and a sequence-to-sequence model for action enumeration. The system proves 14% of its test theorems from Metamath's set.mm module.
     </details>

69. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** [[pdf]](http://arxiv.org/abs/2308.09687) `AAAI 2024 Natural Language Processing` (335 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph of Thoughts is introduced: a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts, and is ensured that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Graph of Thoughts (GoT): a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts (ToT). The key idea and primary advantage of GoT is the ability to model the information generated by an LLM as an arbitrary graph, where units of information ("LLM thoughts") are vertices, and edges correspond to dependencies between these vertices. This approach enables combining arbitrary LLM thoughts into synergistic outcomes, distilling the essence of whole networks of thoughts, or enhancing thoughts using feedback loops. We illustrate that GoT offers advantages over state of the art on different tasks, for example increasing the quality of sorting by 62% over ToT, while simultaneously reducing costs by >31%. We ensure that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes. This work brings the LLM reasoning closer to human thinking or brain mechanisms such as recurrence, both of which form complex networks
     </details>

70. **Measuring and Narrowing the Compositionality Gap in Language Models** [[pdf]](https://aclanthology.org/2023.findings-emnlp.378) `EMNLP 2023 Findings` (407 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that the single-hop question answering performance improves faster than the multi-hop performance does, therefore the compositionality gap does not decrease, and while more powerful models memorize and recall more factual knowledge, they show no corresponding improvement in their ability to perform compositional reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          We investigate the ability of language models to perform compositional reasoning tasks where the overall solution depends on correctly composing the answers to sub-problems. We measure how often models can correctly answer all sub-problems but not generate the overall solution, a ratio we call the compositionality gap. We evaluate this ratio by asking multi-hop questions with answers that require composing multiple facts unlikely to have been observed together during pretraining. In the GPT-3 family of models, as model size increases we show that the single-hop question answering performance improves faster than the multi-hop performance does, therefore the compositionality gap does not decrease. This surprising result suggests that while more powerful models memorize and recall more factual knowledge, they show no corresponding improvement in their ability to perform this kind of compositional reasoning. We then demonstrate how elicitive prompting (such as chain of thought) narrows the compositionality gap by reasoning explicitly instead of implicitly. We present a new method, self-ask, that further improves on chain of thought. In our method, the model explicitly asks itself (and then answers) follow-up questions before answering the initial question. We finally show that self-ask’s structured prompting lets us easily plug in a search engine to answer the follow-up questions, which additionally improves accuracy.
     </details>

71. **Progressive-Hint Prompting Improves Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2304.09797) `2023-08-09` (86 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a new prompting method, named Progressive-Hint Prompting (PHP), that enables automatic multiple interactions between users and LLMs by using previously generated answers as hints to progressively guide toward the correct answers.
     </details>


     <details>
          <summary>Abstract</summary>
          The performance of Large Language Models (LLMs) in reasoning tasks depends heavily on prompt design, with Chain-of-Thought (CoT) and self-consistency being critical methods that enhance this ability. However, these methods do not fully exploit the answers generated by the LLM to guide subsequent responses. This paper proposes a new prompting method, named Progressive-Hint Prompting (PHP), that enables automatic multiple interactions between users and LLMs by using previously generated answers as hints to progressively guide toward the correct answers. PHP is orthogonal to CoT and self-consistency, making it easy to combine with state-of-the-art techniques to further improve performance. We conducted extensive and comprehensive experiments on seven benchmarks. The results show that PHP significantly improves accuracy while remaining highly efficient. For instance, with text-davinci-003, we observed a 4.2% improvement on GSM8K with greedy decoding compared to Complex CoT, and a 46.17% reduction in sample paths with self-consistency. With GPT-4 and PHP, we achieve state-of-the-art performances on SVAMP (89.1% -> 91.9%), GSM8K (92% -> 95.5%), AQuA (76.4% -> 79.9%) and MATH (50.3% -> 53.9%).
     </details>

72. **Specializing Smaller Language Models towards Multi-Step Reasoning** [[pdf]](https://proceedings.mlr.press/v202/fu23d.html) `ICML 2023 Oral` (177 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows two important aspects of model abilities: there exists a very complex balance/ tradeoff between language models' multi-dimensional abilities, and by paying the price of decreased generic ability, it can clearly lift up the scaling curve of models smaller than 10B towards a specialized multi-step math reasoning ability.
     </details>


     <details>
          <summary>Abstract</summary>
          The surprising ability of Large Language Models (LLMs) to perform well on complex reasoning with only few-shot chain-of-thought prompts is believed to emerge only in very large-scale models. We show that such abilities can, in fact, be distilled down from GPT-3.5 (≥ 175B) to T5 variants (≤ 11B). We propose model specialization, to specialize the model’s ability towards a target task. The hypothesis is that large models (commonly viewed as larger than 100B) have strong modeling power such that they can perform a large spectrum of tasks. Small models (commonly viewed as smaller than 10B) have limited model capacity, but if we specialize their capacity towards a target task, the model can achieve decent performance improvements. We use multi-step math reasoning as our testbed because it is a very typical emergent ability. We show two important aspects of model abilities: (1) balancing language model’s performance on multiple tasks is a delicate matter, as improvements on one task may compromise other tasks; (2) yet by intentionally paying the price of decreased generic ability, we can clearly improve across different model scales smaller than 10B towards a specialized multi-step math reasoning ability. We further give comprehensive discussions about important design choices for better generalization, including the data format mixture and the start model checkpoint. We hope our practice and discoveries can serve as an important attempt towards specialized smaller models in the new research paradigm set by LLMs.
     </details>

73. **Automatic Chain of Thought Prompting in Large Language Models** [[pdf]](http://arxiv.org/abs/2210.03493) `ICLR 2023 Poster` (403 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An automatic CoT prompting method that samples questions with diversity and generates reasoning chains to construct demonstrations and consistently matches or exceeds the performance of the CoT paradigm that requires manual designs of demonstrations.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) can carry out complex reasoning tasks by generating intermediate reasoning steps. These steps are triggered by what is called chain-of-thought (CoT) prompting, which comes in two flavors: one leverages a simple prompt like "Let’s think step by step" to facilitate step-by-step reasoning before answering a question (Zero-Shot-CoT). The other uses manual demonstrations, each composed of a question and a reasoning chain that leads to an answer (Manual-CoT). Unfortunately, the superior performance of the latter strategy crucially hinges on manually generating task-specific demonstrations. This makes it far less scalable and more dependent on the talent of the CoT engineer. We show that such manual efforts may be eliminated by leveraging LLMs to generate the reasoning chains on its own. Since these generated chains often come with mistakes we propose a number of mitigation strategies. Our proposed Auto-CoT method automaticaly samples diverse questions and we perform post-processing quality control to generate usable reasoning chains from Zero-Shot-CoT. On ten public benchmark reasoning tasks, Auto-CoT performs on par with Manual-CoT without the need for human intervention. Code is available at https://github.com/amazon-research/auto-cot.
     </details>

74. **Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning** [[pdf]](http://arxiv.org/abs/2205.09712) `ICLR 2023 Notable-top-5%25` (275 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Selection-Inference (SI) framework is proposed that exploits pre-trained LLMs as general processing modules, and alternates between selection and inference to generate a series of interpretable, casual reasoning steps leading to the final answer.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been shown to be capable of impressive few-shot generalisation to new tasks. However, they still tend to perform poorly on multi-step logical reasoning problems. Here we carry out a comprehensive evaluation of LLMs on 46 tasks that probe different aspects of logical reasoning. We show that language models tend to perform fairly well at single step inference or entailment tasks, but struggle to chain together multiple reasoning steps to solve more complex problems. In light of this, we propose a Selection-Inference (SI) framework that exploits pre-trained LLMs as general processing modules, and alternates between selection and inference to generate a series of interpretable, casual reasoning steps leading to the final answer. We show that a 7B parameter LLM used within the SI framework in a 5-shot generalisation setting, with no fine-tuning, yields a performance improvement of over 100% compared to an equivalent vanilla baseline on a suite of 10 logical reasoning tasks. The same model in the same setting even outperforms a significantly larger 280B parameter baseline on the same suite of tasks. Moreover, answers produced by the SI framework are accompanied by a causal natural-language-based reasoning trace, which has important implications for the safety and trustworthiness of the system.
     </details>

75. **Neural Math Word Problem Solver with Reinforcement Learning** [[pdf]](https://www.semanticscholar.org/paper/Neural-Math-Word-Problem-Solver-with-Reinforcement-Huang-Liu/caeb950e503872a903e18a3b259424e3cc3c6006) `2018-08-01` (75 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results show that the copy and alignment mechanism is effective to address the two issues and Reinforcement learning leads to better performance than maximum likelihood on this task; and the neural model is complementary to the feature-based model and their combination significantly outperforms the state-of-the-art results.
     </details>


     <details>
          <summary>Abstract</summary>
          Sequence-to-sequence model has been applied to solve math word problems. The model takes math problem descriptions as input and generates equations as output. The advantage of sequence-to-sequence model requires no feature engineering and can generate equations that do not exist in training data. However, our experimental analysis reveals that this model suffers from two shortcomings: (1) generate spurious numbers; (2) generate numbers at wrong positions. In this paper, we propose incorporating copy and alignment mechanism to the sequence-to-sequence model (namely CASS) to address these shortcomings. To train our model, we apply reinforcement learning to directly optimize the solution accuracy. It overcomes the “train-test discrepancy” issue of maximum likelihood estimation, which uses the surrogate objective of maximizing equation likelihood during training while the evaluation metric is solution accuracy (non-differentiable) at test time. Furthermore, to explore the effectiveness of our neural model, we use our model output as a feature and incorporate it into the feature-based model. Experimental results show that (1) The copy and alignment mechanism is effective to address the two issues; (2) Reinforcement learning leads to better performance than maximum likelihood on this task; (3) Our neural model is complementary to the feature-based model and their combination significantly outperforms the state-of-the-art results.
     </details>

76. **MathDQN: Solving Arithmetic Word Problems via Deep Reinforcement Learning** [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/11981) `AAAI 2018` (106 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This is the first attempt of applying deep reinforcement learning to solve arithmetic word problems and yields remarkable improvement on most of datasets and boosts the average precision among all the benchmark datasets by 15\%.
     </details>


     <details>
          <summary>Abstract</summary>
          Designing an automatic solver for math word problems has been considered as a crucial step towards general AI, with the ability of natural language understanding and logical inference. The state-of-the-art performance was achieved by enumerating all the possible expressions from the quantities in the text and customizing a scoring function to identify the one with the maximum probability. However, it incurs exponential search space with the number of quantities and beam search has to be applied to trade accuracy for efficiency. In this paper, we make the first attempt of applying deep reinforcement learning to solve arithmetic word problems. The motivation is that deep Q-network has witnessed success in solving various problems with big search space and achieves promising performance in terms of both accuracy and running time. To fit the math problem scenario, we propose our MathDQN that is customized from the general deep reinforcement learning framework. Technically, we design the states, actions, reward function, together with a feed-forward neural network as the deep Q-network. Extensive experimental results validate our superiority over state-of-the-art methods. Our MathDQN yields remarkable improvement on most of datasets and boosts the average precision among all the benchmark datasets by 15\%.
     </details>

77. **Large Language Models Cannot Self-Correct Reasoning Yet** [[pdf]](https://openreview.net/forum?id=IkmD3fKBPQ) `ICLR 2024 Poster` (236 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is indicated that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance might even degrade post self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.
     </details>

78. **MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models** [[pdf]](http://arxiv.org/abs/2309.12284) `ICLR 2024 Spotlight` (163 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on two popular benchmarks for mathematical reasoning demonstrate that MetaMath outperforms a suite of open-source LLMs by a significant margin.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have pushed the limits of natural language understanding and exhibited excellent problem-solving ability. Despite the great success, most existing open-source LLMs (\eg, LLaMA-2) are still far away from satisfactory for solving mathematical problems due to the complex reasoning procedures. To bridge this gap, we propose \emph{MetaMath}, a finetuned language model that specializes in mathematical reasoning. Specifically, we start by bootstrapping mathematical questions by rewriting the question from multiple perspectives, which results in a new dataset called {MetaMathQA}. Then we finetune the LLaMA-2 models on MetaMathQA. Experimental results on two popular benchmarks (\ie, GSM8K and MATH) for mathematical reasoning demonstrate that MetaMath outperforms a suite of open-source LLMs by a significant margin. Our MetaMath-7B model achieves $66.5\%$ on GSM8K and $19.8\%$ on MATH, exceeding the state-of-the-art models of the same size by $11.5\%$ and $8.7\%$. Particularly, MetaMath-70B achieves an accuracy of $82.3\%$ on GSM8K, slightly better than GPT-3.5-Turbo. We release the MetaMathQA dataset, the MetaMath models with different model sizes and the training code for public use.
     </details>

79. **Learning to Prove Theorems by Learning to Generate Theorems** [[pdf]](https://arxiv.org/abs/2002.07019) `NeurIPS 2020` `Holophrasm` (41 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover, and demonstrates that synthetic data from this approach improves the theorem provers and advances the state of the art of automated theorem proving in Metamath.
     </details>


     <details>
          <summary>Abstract</summary>
          We consider the task of automated theorem proving, a key AI task. Deep learning has shown promise for training theorem provers, but there are limited human-written theorems and proofs available for supervised learning. To address this limitation, we propose to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover. Experiments on real-world tasks demonstrate that synthetic data from our approach improves the theorem prover and advances the state of the art of automated theorem proving in Metamath. Code is available at https://github.com/princeton-vl/MetaGen.
     </details>

80. **Modeling Intra-Relation in Math Word Problems with Different Functional Multi-Head Attentions** [[pdf]](https://www.aclweb.org/anthology/P19-1619) `ACL 2019 Main` (89 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The experimental results show that the proposed approach performs significantly better than previous state-of-the-art methods, and boost performance from 66.8% to 66.9% on Math23K with 5-fold cross-validation and from 69.2% to 76.1% on MAWPS.
     </details>


     <details>
          <summary>Abstract</summary>
          Several deep learning models have been proposed for solving math word problems (MWPs) automatically. Although these models have the ability to capture features without manual efforts, their approaches to capturing features are not specifically designed for MWPs. To utilize the merits of deep learning models with simultaneous consideration of MWPs’ specific features, we propose a group attention mechanism to extract global features, quantity-related features, quantity-pair features and question-related features in MWPs respectively. The experimental results show that the proposed approach performs significantly better than previous state-of-the-art methods, and boost performance from 66.9% to 69.5% on Math23K with training-test split, from 65.8% to 66.9% on Math23K with 5-fold cross-validation and from 69.2% to 76.1% on MAWPS.
     </details>

81. **Reinforcement Learning of Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/920febb03475b068286a855c10ea09b968fe7ee3) `NeurIPS 2018` `Mizar` (135 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A theorem proving algorithm that uses practically no domain heuristics for guiding its connection-style proof search and solves within the same number of inferences over 40% more problems than a baseline prover, which is an unusually high improvement in this hard AI domain.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

82. **Unit Dependency Graph and Its Application to Arithmetic Word Problem Solving** [[pdf]](http://arxiv.org/abs/1612.00969) `AAAI 2017 NLP and Knowledge Representation` (91 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A decomposed model for inducing UDGs with minimal additional annotations is introduced, and it is shown that introduction of UDGs reduces the error of the solver by over 10 %, surpassing all existing systems for solving arithmetic word problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Math word problems provide a natural abstraction to a range of natural language understanding problems that involve reasoning about quantities, such as interpreting election results, news about casualties, and the financial section of a newspaper. Units associated with the quantities often provide information that is essential to support this reasoning. This paper proposes a principled way to capture and reason about units and shows how it can benefit an arithmetic word problem solver. This paper presents the concept of Unit Dependency Graphs (UDGs), which provides a compact representation of the dependencies between units of numbers mentioned in a given problem. Inducing the UDG alleviates the brittleness of the unit extraction system and allows for a natural way to leverage domain knowledge about unit compatibility, for word problem solving. We introduce a decomposed model for inducing UDGs with minimal additional annotations, and use it to augment the expressions used in the arithmetic word problem solver of (Roy and Roth 2015) via a constrained inference framework. We show that introduction of UDGs reduces the error of the solver by over 10 %, surpassing all existing systems for solving arithmetic word problems. In addition, it also makes the system more robust to adaptation to new vocabulary and equation forms .
     </details>

83. **Learning To Use Formulas To Solve Simple Arithmetic Problems** [[pdf]](https://aclanthology.org/P16-1202) `ACL 2016 Long Papers` (106 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method to learn to use formulas to solve simple arithmetic word problems and beats the state-of-the-art by 86.07% of the problems in a corpus of standard primary school test questions.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

84. **Llemma: An Open Language Model for Mathematics** [[pdf]](http://arxiv.org/abs/2310.10631) `ICLR 2024` `Lean, Isabelle` (164 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Llemma is a large language model for mathematics that outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis, and is capable of tool use and formal theorem proving without any further finetuning.
     </details>


     <details>
          <summary>Abstract</summary>
          We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known openly released models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments.
     </details>

85. **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models** [[pdf]](http://arxiv.org/abs/2306.15626) `NeurIPS 2023` `Lean` (111 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks, and develops ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selection—a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.
     </details>

86. **Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning** [[pdf]](http://arxiv.org/abs/2105.04165) `ACL 2021 Long Papers` (126 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work constructs a new largescale benchmark, Geometry3K, consisting of 3,002 geometry problems with dense annotation in formal language, and proposes a novel geometry solving approach with formal language and symbolic reasoning, called Interpretable Geometry Problem Solver (InterGPS).
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving has attracted much attention in the NLP community recently. The task is challenging as it requires abstract problem understanding and symbolic reasoning with axiomatic knowledge. However, current datasets are either small in scale or not publicly available. Thus, we construct a new large-scale benchmark, Geometry3K, consisting of 3,002 geometry problems with dense annotation in formal language. We further propose a novel geometry solving approach with formal language and symbolic reasoning, called Interpretable Geometry Problem Solver (Inter-GPS). Inter-GPS first parses the problem text and diagram into formal language automatically via rule-based text parsing and neural object detecting, respectively. Unlike implicit learning in existing methods, Inter-GPS incorporates theorem knowledge as conditional rules and performs symbolic reasoning step by step. Also, a theorem predictor is designed to infer the theorem application sequence fed to the symbolic solver for the more efficient and reasonable searching path. Extensive experiments on the Geometry3K and GEOS datasets demonstrate that Inter-GPS achieves significant improvements over existing methods. The project with code and data is available at https://lupantech.github.io/inter-gps.
     </details>

87. **Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems** [[pdf]](http://arxiv.org/abs/2010.06823) `EMNLP 2020 Main` (58 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple but efficient method to make the first attempt to represent the equations of various MWPs uniformly, and a semantically-aligned universal tree-structured solver (SAU-Solver) based on an encoder-decoder framework is proposed to resolve multiple types of MWPs in a unified model, benefiting from the UET representation.
     </details>


     <details>
          <summary>Abstract</summary>
          A practical automatic textual math word problems (MWPs) solver should be able to solve various textual MWPs while most existing works only focused on one-unknown linear MWPs. Herein, we propose a simple but efficient method called Universal Expression Tree (UET) to make the first attempt to represent the equations of various MWPs uniformly. Then a semantically-aligned universal tree-structured solver (SAU-Solver) based on an encoder-decoder framework is proposed to resolve multiple types of MWPs in a unified model, benefiting from our UET representation. Our SAU-Solver generates a universal expression tree explicitly by deciding which symbol to generate according to the generated symbols’ semantic meanings like human solving MWPs. Besides, our SAU-Solver also includes a novel subtree-level semanticallyaligned regularization to further enforce the semantic constraints and rationality of the generated expression tree by aligning with the contextual information. Finally, to validate the universality of our solver and extend the research boundary of MWPs, we introduce a new challenging Hybrid Math Word Problems dataset (HMWP), consisting of three types of MWPs. Experimental results on several MWPs datasets show that our model can solve universal types of MWPs and outperforms several state-of-the-art models.
     </details>

88. **Premise Selection for Theorem Proving by Deep Graph Embedding** [[pdf]](http://arxiv.org/abs/1709.09994) `NeurIPS 2017` (124 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A deep learning-based approach to the problem of premise selection: selecting mathematical statements relevant for proving a given conjecture by representing a higher-order logic formula as a graph that is invariant to variable renaming but still fully preserves syntactic and semantic information.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a deep learning-based approach to the problem of premise selection: selecting mathematical statements relevant for proving a given conjecture. We represent a higher-order logic formula as a graph that is invariant to variable renaming but still fully preserves syntactic and semantic information. We then embed the graph into a vector via a novel embedding method that preserves the information of edge ordering. Our approach achieves state-of-the-art results on the HolStep dataset, improving the classification accuracy from 83% to 90.3%.
     </details>

89. **Solving Geometry Problems: Combining Text and Diagram Interpretation** [[pdf]](https://aclanthology.org/D15-1171) `2015-09-01` (149 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GEOS is introduced, the first automated system to solve unaltered SAT geometry questions by combining text understanding and diagram interpretation, and it is shown that by integrating textual and visual information, GEOS boosts the accuracy of dependency and semantic parsing of the question text.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

90. **Making Language Models Better Reasoners with Step-Aware Verifier** [[pdf]](https://aclanthology.org/2023.acl-long.291) `ACL 2023 Long Papers` (138 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents DiVeRSe (Diverse Verifier on Reasoning Step), a novel approach that further enhances the reasoning capability of language models and achieves new state-of-the-art results on six of eight reasoning benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Few-shot learning is a challenging task that requires language models to generalize from limited examples. Large language models like GPT-3 and PaLM have made impressive progress in this area, but they still face difficulties in reasoning tasks such as GSM8K, a benchmark for arithmetic problems. To improve their reasoning skills, previous work has proposed to guide the language model with prompts that elicit a series of reasoning steps before giving the final answer, achieving a significant improvement on GSM8K from 17.9% to 58.1% in problem-solving rate. In this paper, we present DiVeRSe (Diverse Verifier on Reasoning Step), a novel approach that further enhances the reasoning capability of language models. DiVeRSe has three main components: first, it generates diverse prompts to explore different reasoning paths for the same question; second, it uses a verifier to filter out incorrect answers based on a weighted voting scheme; and third, it verifies each reasoning step individually instead of the whole chain. We evaluate DiVeRSe on the latest language model code-davinci-002 and show that it achieves new state-of-the-art results on six of eight reasoning benchmarks (e.g., GSM8K 74.4% to 83.2%).
     </details>

91. **Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction** [[pdf]](http://arxiv.org/abs/2203.10316) `ACL 2022 Long Papers` (64 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work views the task as a complex relation extraction problem, and proposes a novel approach that presents explainable deductive reasoning steps to iteratively construct target expressions, where each step involves a primitive operation over two quantities defining their relation.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving math word problems requires deductive reasoning over the quantities in the text. Various recent research efforts mostly relied on sequence-to-sequence or sequence-to-tree models to generate mathematical expressions without explicitly performing relational reasoning between quantities in the given context. While empirically effective, such approaches typically do not provide explanations for the generated expressions. In this work, we view the task as a complex relation extraction problem, proposing a novel approach that presents explainable deductive reasoning steps to iteratively construct target expressions, where each step involves a primitive operation over two quantities defining their relation. Through extensive experiments on four benchmark datasets, we show that the proposed model significantly outperforms existing strong baselines. We further demonstrate that the deductive procedure not only presents more explainable steps but also enables us to make more accurate predictions on questions that require more complex reasoning.
     </details>

92. **NumGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2204.05660) `ACL 2022 Long Papers` (91 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          NumGLUE is proposed, a multi-task benchmark that evaluates the performance of AI systems on eight different tasks, that at their core require simple arithmetic understanding and it is shown that this benchmark is far from being solved with neural models including state-of-the-art large-scale language models performing significantly worse than humans.
     </details>


     <details>
          <summary>Abstract</summary>
          Given the ubiquitous nature of numbers in text, reasoning with numbers to perform simple calculations is an important skill of AI systems. While many datasets and models have been developed to this end, state-of-the-art AI systems are brittle; failing to perform the underlying mathematical reasoning when they appear in a slightly different scenario. Drawing inspiration from GLUE that was proposed in the context of natural language understanding, we propose NumGLUE, a multi-task benchmark that evaluates the performance of AI systems on eight different tasks, that at their core require simple arithmetic understanding. We show that this benchmark is far from being solved with neural models including state-of-the-art large-scale language models performing significantly worse than humans (lower by 46.4 %). Further, NumGLUE promotes sharing knowledge across tasks, especially those with limited training data as evidenced by the superior performance (average gain of 3.4 % on each task) when a model is jointly trained on all the tasks as opposed to task-specific modeling. Finally, we hope that NumGLUE will encourage systems that perform robust and general arithmetic reasoning within language, a first step towards being able to perform more complex mathematical reasoning.
     </details>

93. **Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems** [[pdf]](http://arxiv.org/abs/2110.08464) `ACL 2022 Findings` (49 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper investigates how a neural network understands patterns only from semantics, and proposes a contrastive learning approach, where the neural network perceives the divergence of patterns.
     </details>


     <details>
          <summary>Abstract</summary>
          Math Word Problem (MWP) solving needs to discover the quantitative relationships over natural language narratives. Recent work shows that existing models memorize procedures from context and rely on shallow heuristics to solve MWPs. In this paper, we look at this issue and argue that the cause is a lack of overall understanding of MWP patterns. We first investigate how a neural network understands patterns only from semantics, and observe that, if the prototype equations are the same, most problems get closer representations and those representations apart from them or close to other prototypes tend to produce wrong solutions. Inspired by it, we propose a contrastive learning approach, where the neural network perceives the divergence of patterns. We collect contrastive examples by converting the prototype equation into a tree and seeking similar tree structures. The solving model is trained with an auxiliary objective on the collected examples, resulting in the representations of problems with similar prototypes being pulled closer. We conduct experiments on the Chinese dataset Math23k and the English dataset MathQA. Our method greatly improves the performance in monolingual and multilingual settings.
     </details>

94. **Graph-to-Tree Neural Networks for Learning Structured Input-Output Translation with Applications to Semantic Parsing and Math Word Problem** [[pdf]](https://aclanthology.org/2020.findings-emnlp.255) `EMNLP 2020 Findings` (64 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a novel Graph-to-Tree Neural Networks, namely Graph2Tree consisting of a graph encoder and a hierarchical tree decoder, that encodes an augmented graph-structured input and decodes a tree- Structured output.
     </details>


     <details>
          <summary>Abstract</summary>
          The celebrated Seq2Seq technique and its numerous variants achieve excellent performance on many tasks such as neural machine translation, semantic parsing, and math word problem solving. However, these models either only consider input objects as sequences while ignoring the important structural information for encoding, or they simply treat output objects as sequence outputs instead of structural objects for decoding. In this paper, we present a novel Graph-to-Tree Neural Networks, namely Graph2Tree consisting of a graph encoder and a hierarchical tree decoder, that encodes an augmented graph-structured input and decodes a tree-structured output. In particular, we investigated our model for solving two problems, neural semantic parsing and math word problem. Our extensive experiments demonstrate that our Graph2Tree model outperforms or matches the performance of other state-of-the-art models on these tasks.
     </details>

95. **Graph Representations for Higher-Order Logic and Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/4b127897595af5a97c83860eec0540de5510f646) `AAAI 2020 Knowledge Representation and Reasoning` (87 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents the first use of graph neural networks (GNNs) for higher-order proof search and demonstrates that GNNs can improve upon state-of-the-art results in this domain.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents the first use of graph neural networks (GNNs) for higher-order proof search and demonstrates that GNNs can improve upon state-of-the-art results in this domain. Interactive, higher-order theorem provers allow for the formalization of most mathematical theories and have been shown to pose a significant challenge for deep learning. Higher-order logic is highly expressive and, even though it is well-structured with a clearly defined grammar and semantics, there still remains no well-established method to convert formulas into graph-based representations. In this paper, we consider several graphical representations of higher-order logic and evaluate them against the HOList benchmark for higher-order theorem proving.
     </details>

96. **Tree-structured Decoding for Solving Math Word Problems** [[pdf]](https://www.aclweb.org/anthology/D19-1241) `EMNLP 2019 Main` (81 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A tree-structured decoding method that generates the abstract syntax tree of the equation in a top-down manner and can automatically stop during decoding without a redundant stop token is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Automatically solving math word problems is an interesting research topic that needs to bridge natural language descriptions and formal math equations. Previous studies introduced end-to-end neural network methods, but these approaches did not efficiently consider an important characteristic of the equation, i.e., an abstract syntax tree. To address this problem, we propose a tree-structured decoding method that generates the abstract syntax tree of the equation in a top-down manner. In addition, our approach can automatically stop during decoding without a redundant stop token. The experimental results show that our method achieves single model state-of-the-art performance on Math23K, which is the largest dataset on this task.
     </details>

97. **Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems** [[pdf]](https://aclanthology.org/E17-1047) `2017-04-01` (51 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new evaluation for automatic solvers for algebra word problems is proposed, which can identify mistakes that existing evaluations overlook, and derivation annotations can be semi-automatically added to existing datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a new evaluation for automatic solvers for algebra word problems, which can identify mistakes that existing evaluations overlook. Our proposal is to evaluate such solvers using derivations, which reflect how an equation system was constructed from the word problem. To accomplish this, we develop an algorithm for checking the equivalence between two derivations, and show how derivation annotations can be semi-automatically added to existing datasets. To make our experiments more comprehensive, we include the derivation annotation for DRAW-1K, a new dataset containing 1000 general algebra word problems. In our experiments, we found that the annotated derivations enable a more accurate evaluation of automatic solvers than previously used metrics. We release derivation annotations for over 2300 algebra word problems for future evaluations.
     </details>

98. **TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance** [[pdf]](http://arxiv.org/abs/2105.07624) `ACL 2021 Long Papers` (200 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work extracts samples from real financial reports to build a new large-scale QA dataset containing both Tabular And Textual data, named TAT-QA, where numerical reasoning is usually required to infer the answer, such as addition, subtraction, multiplication, division, counting, comparison/sorting, and the compositions.
     </details>


     <details>
          <summary>Abstract</summary>
          Hybrid data combining both tabular and textual content (e.g., financial reports) are quite pervasive in the real world. However, Question Answering (QA) over such hybrid data is largely neglected in existing research. In this work, we extract samples from real financial reports to build a new large-scale QA dataset containing both Tabular And Textual data, named TAT-QA, where numerical reasoning is usually required to infer the answer, such as addition, subtraction, multiplication, division, counting, comparison/sorting, and the compositions. We further propose a novel QA model termed TAGOP, which is capable of reasoning over both tables and text. It adopts sequence tagging to extract relevant cells from the table along with relevant spans from the text to infer their semantics, and then applies symbolic reasoning over them with a set of aggregation operators to arrive at the final answer. TAGOP achieves 58.0% inF1, which is an 11.1% absolute increase over the previous best baseline model, according to our experiments on TAT-QA. But this result still lags far behind performance of expert human, i.e.90.8% in F1. It is demonstrated that our TAT-QA is very challenging and can serve as a benchmark for training and testing powerful QA models that address hybrid form data.
     </details>

99. **A Knowledge-Aware Sequence-to-Tree Network for Math Word Problem Solving** [[pdf]](https://aclanthology.org/2020.emnlp-main.579) `EMNLP 2020 Main` (62 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results revealed that the KA-S2T model can achieve better performance than previously reported best results and use a tree-structured decoder with a state aggregation mechanism to capture the long-distance dependency and global expression information.
     </details>


     <details>
          <summary>Abstract</summary>
          With the advancements in natural language processing tasks, math word problem solving has received increasing attention. Previous methods have achieved promising results but ignore background common-sense knowledge not directly provided by the problem. In addition, during generation, they focus on local features while neglecting global information. To incorporate external knowledge and global expression information, we propose a novel knowledge-aware sequence-to-tree (KA-S2T) network in which the entities in the problem sequences and their categories are modeled as an entity graph. Based on this entity graph, a graph attention network is used to capture knowledge-aware problem representations. Further, we use a tree-structured decoder with a state aggregation mechanism to capture the long-distance dependency and global expression information. Experimental results on the Math23K dataset revealed that the KA-S2T model can achieve better performance than previously reported best results.
     </details>

100. **Injecting Numerical Reasoning Skills into Language Models** [[pdf]](http://arxiv.org/abs/2004.04487) `ACL 2020 Main` (205 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows that numerical reasoning is amenable to automatic data generation, and thus one can inject this skill into pre-trained LMs, by generating large amounts of data, and training in a multi-task setup.
     </details>


     <details>
          <summary>Abstract</summary>
          Large pre-trained language models (LMs) are known to encode substantial amounts of linguistic information. However, high-level reasoning skills, such as numerical reasoning, are difficult to learn from a language-modeling objective only. Consequently, existing models for numerical reasoning have used specialized architectures with limited flexibility. In this work, we show that numerical reasoning is amenable to automatic data generation, and thus one can inject this skill into pre-trained LMs, by generating large amounts of data, and training in a multi-task setup. We show that pre-training our model, GenBERT, on this data, dramatically improves performance on DROP (49.3 –> 72.3 F1), reaching performance that matches state-of-the-art models of comparable size, while using a simple and general-purpose encoder-decoder architecture. Moreover, GenBERT generalizes well to math word problem datasets, while maintaining high performance on standard RC tasks. Our approach provides a general recipe for injecting skills into large pre-trained LMs, whenever the skill is amenable to automatic data augmentation.
     </details>

101. **First Neural Conjecturing Datasets and Experiments** [[pdf]](https://arxiv.org/abs/2005.14664) `2020-01-01` (34 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Several datasets and first experiments with creating conjectures by neural methods based on the Mizar Mathematical Library processed in several forms and the problems extracted by the MPTP system and proved by the E prover using the ENIGMA guidance are described.
     </details>


     <details>
          <summary>Abstract</summary>
          We describe several datasets and first experiments with creating conjectures by neural methods. The datasets are based on the Mizar Mathematical Library processed in several forms and the problems extracted from it by the MPTP system and proved by the E prover using the ENIGMA guidance. The conjecturing experiments use the Transformer architecture and in particular its GPT-2 implementation.
     </details>

102. **Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension** [[pdf]](https://www.aclweb.org/anthology/D19-1609) `EMNLP 2019 Main` (96 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work enables a BERT-based reading comprehension model to perform lightweight numerical reasoning by augmenting the model with a predefined set of executable ‘programs’ which encompass simple arithmetic as well as extraction.
     </details>


     <details>
          <summary>Abstract</summary>
          Reading comprehension models have been successfully applied to extractive text answers, but it is unclear how best to generalize these models to abstractive numerical answers. We enable a BERT-based reading comprehension model to perform lightweight numerical reasoning. We augment the model with a predefined set of executable ‘programs’ which encompass simple arithmetic as well as extraction. Rather than having to learn to manipulate numbers directly, the model can pick a program and execute it. On the recent Discrete Reasoning Over Passages (DROP) dataset, designed to challenge reading comprehension models, we show a 33% absolute improvement by adding shallow programs. The model can learn to predict new operations when appropriate in a math word problem setting (Roy and Roth, 2015) with very few training examples.
     </details>

103. **HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving** [[pdf]](http://arxiv.org/abs/1703.00426) `ICLR 2017` `HOL Light` (79 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new dataset based on Higher-Order Logic (HOL) proofs is introduced, for the purpose of developing new machine learning-based theorem-proving strategies and the results of these models show the promise of applying machine learning to HOL theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          Large computer-understandable proofs consist of millions of intermediate logical steps. The vast majority of such steps originate from manually selected and manually guided heuristics applied to intermediate goals. So far, machine learning has generally not been used to filter or generate these steps. In this paper, we introduce a new dataset based on Higher-Order Logic (HOL) proofs, for the purpose of developing new machine learning-based theorem-proving strategies. We make this dataset publicly available under the BSD license. We propose various machine learning tasks that can be performed on this dataset, and discuss their significance for theorem proving. We also benchmark a set of simple baseline machine learning models suited for the tasks (including logistic regression, convolutional neural networks and recurrent neural networks). The results of our baseline models show the promise of applying machine learning to HOL theorem proving.
     </details>

