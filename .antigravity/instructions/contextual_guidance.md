# Instruction: Explain Why In Guidance

When giving implementation guidance, include a brief explanation of why each recommended step exists.

- For each actionable step, add 1-2 sentences of context on purpose, failure mode avoided, or expected outcome.
- Prefer concise rationale over long theory.
- Keep guidance practical and connected to the current codebase.

# Core Preservation Rule

**Absolute Priority of Logic**: Never prioritize structural, architectural, or compilation fixes at the cost of removing, stripping, or simplifying existing implementation logic. Every line of functional code must be preserved or accurately migrated during any refactor. Skeletal solutions are unacceptable if they result in lost functionality.
