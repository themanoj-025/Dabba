# Design — Dabba: Design System & UX Principles

|Field|Value|
|---|---|
|Version|v0.1|
|Last Updated|2026-08-06|
|Owner|Design Lead|
|Status|In Review|

---

## 1. Design Principles

1. **Appetite first** — ratings and reliability lead every card.
2. **Explainable** — every score links to "why" (narration).
3. **Calm data density** — tables + charts, minimal prose.
4. **Consistent** — token-based colors, shared components.
5. **Honest** — LLM vs template output is labeled.

## 2. Brand & Visual Identity

- Voice: friendly, food-positive, trustworthy.
- Imagery: cuisine/city photography; restaurant cards.

## 3. Color System

|Token|Hex|Usage|Contrast (AA)|
|---|---|---|---|
|bg|`#FFF7ED`|warm background|—|
|surface|`#FFFFFF`|cards|—|
|primary|`#EA580C`|CTAs (food orange)|4.9:1|
|text|`#1C1917`|body|14:1|
|muted|`#78716C`|secondary|4.9:1|
|success|`#16A34A`|reliability high|5.1:1|
|warning|`#D97706`|reliability medium|4.7:1|
|danger|`#DC2626`|reliability low|5.9:1|

## 4. Typography Scale

|Token|Font|Size|Weight|Line-height|Usage|
|---|---|---|---|---|---|
|display|system sans|30px|700|1.2|KPI scores|
|heading|system sans|22px|600|1.3|page titles|
|body|system sans|15px|400|1.5|content|
|caption|system sans|12px|400|1.4|meta|
|score|mono|20px|700|1.2|reliability badges|

## 5. Spacing & Grid

- Base 4px; Streamlit default layout.
- Breakpoints: Streamlit responsive.

## 6. Component Library

**Restaurant card:**

```
┌────────────────────────────────┐
│ [image]  Name — ★ 4.3         │
│          Cuisine · Cost ₹₹     │
│          Reliability: 87/100  │
│          [Why this?] [Similar]│
└────────────────────────────────┘
```

**Reliability badge:** color pill with number (HIGH/MED/LOW text included, not color-only).

Other: KPI card, benchmark table, chat panel, drift alert banner.

## 7. Iconography

Plotly + Unicode emojis (🍽️ ★ ⏱️).

## 8. Accessibility

- WCAG 2.1 AA targets.
- Reliability never conveyed by color alone.
- Keyboard nav for tables + chat.

## 9. Responsive

- Fluid dashboard; tables scroll on small screens.

## 10. Motion

- Chart transitions (300ms); card hover lift; reduced-motion honored.

## 11. Dark Mode

Light theme default; no dark mode in v1.

## 12. Related Documents

|Document|Relationship|
|---|---|
|[AppFlow.md](AppFlow.md)|Screens|
|[PRD.md](../product/PRD.md)|UX goals|
|[TechSpec.md](../technical/TechSpec.md)|Stack|
|[Schema.md](../technical/Schema.md)|Display data|
|[ImplementationPlan.md](../project/ImplementationPlan.md)|Tasks|
|[Tracker.md](../project/Tracker.md)|Status|
|[Rules.md](../project/Rules.md)|Standards|
|[API.md](../technical/API.md)|Contracts|
|[SecurityAndCompliance.md](../technical/SecurityAndCompliance.md)|Access|
|[Testing.md](../technical/Testing.md)|UI tests|
|[Deployment.md](../technical/Deployment.md)|Deploy|
|[Glossary.md](../reference/Glossary.md)|Vocabulary|
|[RiskRegister.md](../project/RiskRegister.md)|Risks|
