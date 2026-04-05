# Rule: ALWAYS Read Context Before Acting

## RULE
Before sending ANY message to ANY tmux agent, you MUST read the agent's pane first (at least -S -20 lines). Understand what the agent is currently doing. Only THEN decide whether to send a message.

## WHY
Multiple times I sent redundant or contradictory messages to agents because I didn't read their context first. This wastes the user's time correcting me, wastes agent context with duplicate instructions, and makes me look impulsive. The user explicitly told me: "read the context then plan the response or we will not be efficient."

## HOW TO APPLY
1. ALWAYS run `tmux capture-pane -t SESSION -p -S -20` BEFORE any `tmux send-keys`
2. READ what the agent is doing, what state it's in, what the user already told it
3. DECIDE: does the agent actually need my message? Is it already doing what I want?
4. If YES → send a targeted message that adds value
5. If NO → don't send anything, report status to user instead
6. NEVER fire off nudges/instructions without reading first

## PATTERN
```
# WRONG (impulsive):
tmux send-keys -t AGENT "do X" C-m

# RIGHT (read first):
tmux capture-pane -t AGENT -p -S -20  # READ
# ... analyze what agent is doing ...
# ... only THEN send if needed ...
tmux send-keys -t AGENT "targeted message" C-m
```

## DO NOT
- Send messages to agents without reading their pane first
- Override what the user already told the agent
- Send redundant instructions (agent is already doing it)
- Assume agent state without checking
