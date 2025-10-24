# Jury Evaluation System - Visual Workflow

## 🏗️ System Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        Q[User Query]
        TC[Test Cases]
        P[System Prompt]
    end
    
    subgraph "RAG System"
        RS[RAG Agent/Chain]
        R[Retriever]
        VDB[(Vector DB)]
        LLM1[Primary LLM]
    end
    
    subgraph "Jury Evaluation System"
        J[Jury Controller]
        E1[Evaluator LLM 1<br/>OpenAI GPT-4]
        E2[Evaluator LLM 2<br/>Llama 70B]
        E3[Evaluator LLM 3<br/>Mistral 7B]
        VS[Voting Strategy]
    end
    
    subgraph "Evaluation Criteria"
        EC1[Empathy 1.3x]
        EC2[Sensitivity 1.3x]
        EC3[Privacy 1.5x]
        EC4[Hallucination 1.2x]
        EC5[Retrieval 1.0x]
        EC6[Completeness 1.0x]
    end
    
    subgraph "Output & Optimization"
        ER[Evaluation Report]
        IS[Improvement Suggestions]
        PO[Prompt Optimizer]
        IA[Improvement Analyzer]
    end
    
    Q --> RS
    RS --> R
    R --> VDB
    VDB --> R
    R --> LLM1
    LLM1 --> RS
    
    RS --> J
    J --> E1
    J --> E2
    J --> E3
    
    E1 --> VS
    E2 --> VS
    E3 --> VS
    
    VS --> EC1
    VS --> EC2
    VS --> EC3
    VS --> EC4
    VS --> EC5
    VS --> EC6
    
    EC1 --> ER
    EC2 --> ER
    EC3 --> ER
    EC4 --> ER
    EC5 --> ER
    EC6 --> ER
    
    ER --> IS
    IS --> PO
    ER --> IA
    PO --> P
    
    style J fill:#e1f5fe
    style VS fill:#f3e5f5
    style ER fill:#e8f5e8
    style PO fill:#fff3e0
```

## 🔄 Detailed Jury Voting Process

```mermaid
sequenceDiagram
    participant U as User Query
    participant RAG as RAG System
    participant J as Jury Controller
    participant E1 as Evaluator 1 (OpenAI)
    participant E2 as Evaluator 2 (Llama)
    participant E3 as Evaluator 3 (Mistral)
    participant VS as Voting Strategy
    participant R as Final Result
    
    U->>RAG: Submit query
    RAG->>RAG: Retrieve documents
    RAG->>RAG: Generate response
    RAG->>J: Response + Context docs
    
    par Parallel Evaluation
        J->>E1: Evaluate criteria (Empathy, Privacy, etc.)
        J->>E2: Evaluate criteria (Empathy, Privacy, etc.)
        J->>E3: Evaluate criteria (Empathy, Privacy, etc.)
    end
    
    E1->>VS: Score: 8.5, Pass, Reasoning, Suggestions
    E2->>VS: Score: 7.2, Pass, Reasoning, Suggestions
    E3->>VS: Score: 6.8, Fail, Reasoning, Suggestions
    
    VS->>VS: Apply weighted voting<br/>(OpenAI: 1.2x, Llama: 1.0x, Mistral: 0.8x)
    VS->>R: Weighted Score: 7.6, Pass, Compiled reasoning
    
    R->>U: Evaluation report + Improvement suggestions
```

## 📊 Evaluation Criteria Weighting System

```mermaid
pie title Evaluation Criteria Weights (Normalized)
    "Privacy Breach" : 13.0
    "Malicious Use" : 13.0
    "Empathy" : 11.3
    "Sensitivity" : 11.3
    "Hallucination" : 10.4
    "Security Breach" : 10.4
    "Retrieval Relevance" : 8.7
    "Noise Robustness" : 8.7
    "Negative Rejection" : 8.7
    "Information Integration" : 8.7
    "Counterfactual Robustness" : 8.7
    "Completeness" : 8.7
    "Out of Domain" : 8.7
    "Brand Damage" : 8.7
```

## 🎯 Voting Strategies Comparison

```mermaid
graph LR
    subgraph "Input Responses"
        R1[LLM1: Score 8.5]
        R2[LLM2: Score 7.2]
        R3[LLM3: Score 6.8]
    end
    
    subgraph "Voting Strategies"
        M[Majority Vote<br/>Most common response]
        W[Weighted Vote<br/>Provider reliability weights]
        U[Unanimous Vote<br/>All must agree]
        A[Average Score<br/>Numerical average]
        F[First Valid<br/>First successful response]
        C[Consensus<br/>Find common themes]
    end
    
    subgraph "Results"
        MR[Result: 7.2<br/>Most frequent]
        WR[Result: 7.6<br/>Weighted average]
        UR[Result: No consensus<br/>Aggregated response]
        AR[Result: 7.5<br/>Simple average]
        FR[Result: 8.5<br/>First response]
        CR[Result: Combined themes]
    end
    
    R1 --> M
    R2 --> M
    R3 --> M
    M --> MR
    
    R1 --> W
    R2 --> W
    R3 --> W
    W --> WR
    
    R1 --> U
    R2 --> U
    R3 --> U
    U --> UR
    
    R1 --> A
    R2 --> A
    R3 --> A
    A --> AR
    
    R1 --> F
    R2 --> F
    R3 --> F
    F --> FR
    
    R1 --> C
    R2 --> C
    R3 --> C
    C --> CR
    
    style W fill:#e8f5e8
    style WR fill:#e8f5e8
```

## 🔧 Complete Optimization Workflow

```mermaid
flowchart TD
    Start([Start Evaluation]) --> Init[Initialize Components<br/>- RAG Evaluator<br/>- Prompt Optimizer<br/>- Improvement Analyzer]
    
    Init --> Eval1[Initial Evaluation<br/>Run test cases through RAG system]
    
    Eval1 --> Check1{Pass Rate >= 80%?}
    Check1 -->|Yes| Success[✅ Optimization Complete]
    Check1 -->|No| Extract[Extract Improvement Suggestions<br/>- Failed criteria analysis<br/>- Specific recommendations]
    
    Extract --> Optimize[Prompt Optimization<br/>Apply suggestions via LLM]
    
    Optimize --> Update[Update System Prompt<br/>Deploy new version]
    
    Update --> Eval2[Re-evaluate<br/>Test with updated prompt]
    
    Eval2 --> Check2{Pass Rate >= 80%<br/>OR Max Iterations?}
    Check2 -->|No| Extract
    Check2 -->|Yes| Analyze[Historical Analysis<br/>- Track improvements<br/>- Identify patterns]
    
    Analyze --> Save[Save Results<br/>- JSON logs<br/>- Summaries<br/>- Optimization history]
    
    Save --> Success
    
    style Check1 fill:#fff3e0
    style Check2 fill:#fff3e0
    style Success fill:#e8f5e8
    style Extract fill:#e1f5fe
    style Optimize fill:#f3e5f5
```

## 📈 Data Flow & Persistence

```mermaid
graph TB
    subgraph "Evaluation Process"
        EP[Evaluation Pipeline]
        JR[Jury Results]
        ER[Evaluation Reports]
    end
    
    subgraph "Storage Layer"
        JSON[JSON Logs<br/>logs/rag_evaluation/json/]
        SUM[Summaries<br/>logs/rag_evaluation/summaries/]
        LLM[LLM Scores<br/>logs/rag_evaluation/llm_scores/]
        OPT[Optimization History<br/>logs/prompt_optimization/]
    end
    
    subgraph "Analysis Layer"
        IA[Improvement Analyzer]
        TR[Trend Reports]
        PS[Priority Suggestions]
    end
    
    subgraph "Optimization Layer"
        PO[Prompt Optimizer]
        IW[Iterative Workflows]
        AP[Automated Prompts]
    end
    
    EP --> JR
    JR --> ER
    
    ER --> JSON
    ER --> SUM
    JR --> LLM
    
    JSON --> IA
    SUM --> IA
    LLM --> IA
    
    IA --> TR
    IA --> PS
    
    PS --> PO
    TR --> PO
    PO --> OPT
    PO --> IW
    IW --> AP
    
    style JSON fill:#e3f2fd
    style IA fill:#f1f8e9
    style PO fill:#fff8e1
```

## 🎮 Interactive Demo Flow

```mermaid
stateDiagram-v2
    [*] --> QuickDemo
    
    state QuickDemo {
        [*] --> InitEvaluator
        InitEvaluator --> LoadTestCase
        LoadTestCase --> RunEvaluation
        RunEvaluation --> DisplayResults
        DisplayResults --> ShowSuggestions
        ShowSuggestions --> [*]
    }
    
    QuickDemo --> EnhancedDemo
    
    state EnhancedDemo {
        [*] --> CreateJury
        CreateJury --> EvaluateMultipleCriteria
        EvaluateMultipleCriteria --> CalculatePassFail
        CalculatePassFail --> GenerateImprovements
        GenerateImprovements --> [*]
    }
    
    EnhancedDemo --> OptimizationWorkflow
    
    state OptimizationWorkflow {
        [*] --> InitialEval
        InitialEval --> ExtractSuggestions
        ExtractSuggestions --> OptimizePrompt
        OptimizePrompt --> ReEvaluate
        ReEvaluate --> CheckThreshold
        CheckThreshold --> OptimizePrompt : Pass Rate < 80%
        CheckThreshold --> Complete : Pass Rate >= 80%
        Complete --> [*]
    }
    
    OptimizationWorkflow --> [*]
```
