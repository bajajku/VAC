'''
- This script is used to evaluate the RAG pipeline.

Summary:
We want to evaluate our Agentic RAG application based on 10 criterias. This evaluation will be an initial step which when combined \
with feedback data from actual sources will be used to guide our prompt engineering, and even finetuning in later stages.

How it works:
- We will use a set of test cases to evaluate the RAG pipeline.
- We will use a set of criterias to evaluate the RAG pipeline.
- We will use a set of feedback data from domain experts who will also rate answers based on the same criterias.

Work flow:
Step-1: Rag Application will be queried with a set of test cases.
Step-2: A Jury of LLM's will be created to evaluate the answers based on the criterias.
Step-3: Jury will deliberate and come up with a final score for each test case. If an RAG application scores below threshold of lets say 85% on all criterias, it will be considered as need to be improved.
Step-4: A feedback report will be generated combining Jury's feedback and human feedback
Step-5: We will give this feedback report to RAG application along with it's system prompt and will ask it improve it's prompt, considering the feedback.
Step-6: Now we will repeat the process again with the same RAG application but with updated system prompt.
Step-7: This process will be repeated until the RAG application scores above threshold of 85% on all criterias, or until 100 iterations are completed.
Step-8: All these scores will be saved along with the system prompt at each iteration, in a final report.
Step-9: This report will be used to guide our prompt engineering, and even finetuning in later stages.

Why this is important:
- This will help us to identify the weak areas of the RAG application.
- This will help us to guide our prompt engineering.
- This will help us to finetune the RAG application in later stages.
- This will help us to identify the best system prompt for the RAG application.
- This will help us to identify the best LLM's for the RAG application.
- This will help us to identify the best data for the RAG application.

'''