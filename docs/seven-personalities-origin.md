# Origin of the Seven Personalities

## Source Paper

This repository's scientific-paper workflow is inspired by the MAPS paper:

- Title: MAPS: A Multi-Agent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving
- Authors: Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Xinyu Zhang, Fangzhi Xu, Qika Lin, Rui Mao, Erik Cambria, Jun Liu
- Venue/Identifier: arXiv:2503.16905v1 [cs.AI]
- Date: 21 Mar 2025

## Why This Matters

MAPS targets Multimodal Scientific Problems (MSPs), where solving requires combining text and visual/diagram information and performing domain-aware reasoning. The paper highlights two core challenges:

1. Multi-modal comprehensive reasoning is hard when a single model must do all steps alone.
2. One-pass reasoning lacks reflection, critique, and iterative correction.

MAPS addresses this with coordinated multi-agent reasoning and Socratic-style feedback loops.

## The Big Seven Personality Basis

The paper grounds agent specialization in the Big Seven personality dimensions:

1. Conscientiousness
2. Agreeableness
3. Extraversion
4. Neuroticism
5. Openness
6. Self-Esteem
7. Sensitivity

In this codebase, these seven dimensions are used as distinct reviewer/synthesis perspectives in the scientific-paper workflow.

## MAPS Idea to Implementation Mapping

MAPS introduces role specialization plus reflective critique. In this project, that inspiration is mapped as follows:

1. Seven personality agents provide distinct critique signals each round.
2. Openness acts as synthesis lead, integrating cross-personality feedback.
3. Conscientiousness acts as a strict approval gate (APPROVED or required fixes).
4. Multi-round interaction enables iterative refinement instead of one-shot output.

This implementation is inspired by MAPS concepts, not a claim of exact reproduction of the original experimental setup.

## Scope in This Repository

The explicit workflow split is:

1. coding workflow: legacy Builder + Editor pipeline
2. scientific-paper workflow: seven-personality pipeline inspired by MAPS

This keeps coding behavior stable while applying personality-driven collaboration to scientific-paper review tasks.

## Citation

If you reference this design origin, cite the MAPS paper:

Zhang, J., Wang, Z., Wang, Z., Zhang, X., Xu, F., Lin, Q., Mao, R., Cambria, E., and Liu, J. (2025). MAPS: A Multi-Agent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving. arXiv:2503.16905v1 [cs.AI].
