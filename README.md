# ML4Mindfulness_Autism

# ML for Mindfulness in Autistic Adults
**Predicting responders vs. non-responders to a smartphone mindfulness intervention using baseline questionnaires**

This repository accompanies the manuscript:

> **Machine learning analysis of smartphone mindfulness interventions in autistic adults: predicting responders and non-responders for anxiety reduction**  
> Gun Ahn, Aixin Liang, Wonchang Choi, Cindy Li, John D. E. Gabrieli

We perform a **secondary analysis** of a randomized controlled trial (RCT) showing that a 6-week, smartphone-based mindfulness program reduced anxiety and perceived stress in autistic adults. Here, we ask: *Can baseline self-report measures predict who benefits most?* We train several ML models with **nested cross-validation**, interpret feature importance (including **SHAP**), and compute an **individualized benefit score** via the **Personalized Advantage Index (PAI)**.

---

## TL;DR (Key Findings)
- **Outcome**: Responder = ≥7-point decrease on STAI-State (post vs. pre), chosen to reflect a clinically meaningful shift across category thresholds.  
- **Sample**: n = 89 randomized (73 app users included in primary analyses; ITT reported separately).  
- **Performance**: Best AUC for state-anxiety response ≈ **0.79** (Random Forest; 95% CI **[0.66, 0.91]**).  
- **Signal**: Baseline **anxiety severity** (STAI-S/T items) dominated importance across models; **age** showed a small negative association with response.  
- **Personalization**: **PAI** identified individuals predicted to benefit more from the app vs. wait-list, illustrating how to move beyond group averages.
