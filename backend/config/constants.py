PROMPT_TEMPLATE = """You are a trauma-informed, empathetic mental health assistant. Your role is to support military personnel and veterans with mental health concerns.

Given the following context and question, provide a helpful response:

Context: {context}
Question: {input}

Guidelines:
- Keep responses between 100-150 words
- Use clear, direct language and avoid repetition 
- Leave blank lines before/after lists
- Begin with most important/helpful point
- Use respectful, gender-neutral language
- Never probe for trauma details
- Validate emotional experiences with empathy

Key Principles:
- Respond with empathy, care and respect
- Use trauma-informed, military-relevant knowledge
- Suggest grounding/mindfulness strategies when appropriate 
- Refer to crisis services if severe distress indicated
- Do not replace therapy - provide supportive not clinical help

Remember:
- Never fabricate or speculate
- Focus on emotional support and trauma-informed practices
- Stay within mental health and veteran support scope
- Do not provide legal/medical advice"""



TEST_CASES = [
    "Why do I still get panic attacks years after deployment?",
    "How do I know if what I’m experiencing is PTSD or just stress?",
    "Is it normal to avoid certain sounds or places after coming back from service?",
    "Why am I suddenly angry at everything even though nothing is happening?",
    "Why do I wake up shaking from nightmares that feel real?",
    "Why do I feel disconnected from my family since returning home?",
    "How do I deal with guilt over things I had to do in combat?",
    "Can someone recover from moral injury or does it stay forever?",
    "I feel like a bad person because of what happened overseas—what do I do?",
    "Why do I feel ashamed even though everyone tells me I did my job?",
    "How do I explain my trauma responses to my partner without sounding weak?",
    "Why do I push away people who care about me?",
    "How do I reconnect with my kids when I feel emotionally numb?",
    "How can I stop myself from snapping at people for small things?",
    "Why do I feel like I lost my sense of purpose after leaving the military?",
    "How do I find motivation when everything feels pointless?",
    "Is it normal to feel empty after transitioning to civilian life?",
    "I’m drinking more than I used to just to sleep—is this a sign of a problem?",
    "How do I manage stress without relying on alcohol?",
    "Why does everything feel worse at night and lead me to unhealthy coping?",
    "Why do sudden noises make my whole body react?",
    "How do I stop replaying the same event in my head?",
    "Is it normal to feel like I’m back in deployment during flashbacks?",
    "What kind of therapy actually works for combat trauma?",
    "Is EMDR helpful for veterans or is CBT better?",
    "I’m scared of therapy because I don’t want to relive everything—what are the alternatives?",
    "How do I start getting help if I don’t trust the system?",
    "What does the VA actually offer for PTSD, and does it work?",
    "Why do I get irritated at things that never bothered me before?",
    "How can I calm myself when my anger spikes out of nowhere?",
    "Is anger a common symptom of combat trauma?",
    "Does having PTSD mean I’m weak?",
    "How do I deal with the feeling that no one understands what I went through?",
    "Why do I feel like a burden to my family?",
    "I’m overwhelmed and don’t know what to do next.",
    "I’m scared of where my thoughts are heading.",
    "I feel like I can’t handle this anymore—what steps can I take right now?",
    "How do I make it through nights when everything feels too heavy?"
]

